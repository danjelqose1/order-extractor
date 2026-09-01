// A fresh process handles one bounded job. No listening socket, network or database access.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'docs/js/platform-workflows.js'), 'utf8');
async function main() {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  const config = request.price_config || {};
  const context = {
    Blob, Uint8Array,
    console: { log() {}, warn() {}, error() {} },
    DIMENSION_TOLERANCE_MM: 1, UNKNOWN_CLIENT_LABEL: '(Unknown Client)',
    pageSize: { w: 100 * 2.83464567, h: 40 * 2.83464567 },
    TYPE_CORRECTIONS: config.typeCorrections || [],
    glassAliasCache: Object.create(null), glassAliasPending: new Map(),
    ensurePdfLib: async () => {},
    preloadLogos: async () => ({
      keliBytes: new Uint8Array(fs.readFileSync(path.join(root, 'docs/logokeli.png'))),
      ceBytes: new Uint8Array(fs.readFileSync(path.join(root, 'docs/ce.png'))),
    }),
    appState: {
      processing: {
        rows: [], rounding: { manualApplied: !!request.rounded },
        options: { autoDanko: false }, preview: request.preview,
      },
      invoices: { priceLists: { glass: config.glassPrices || {}, spacer: config.spacerPrices || {} }, componentAliases: config.componentAliases || {} },
    },
  };
  context.window = context;
  vm.createContext(context);
  // Load PDFLib into the same realm: its array checks use instanceof Array.
  vm.runInContext(fs.readFileSync(require.resolve('pdf-lib/dist/pdf-lib.min.js'), 'utf8'), context, {timeout: 5000});
  vm.runInContext(source, context, {timeout: 5000});
  let result;
  if (request.operation === 'preview') {
    const order = request.order.source === 'manual' ? context.manualOrderToShared(request.order) : request.order;
    const rows = context.convertOrderToProcessingEntry(order).rows;
    context.appState.processing.rows = rows;
    context.applyProcessingRoundingToRows();
    const preview = context.generateMotherSheet(rows, {
      restartPerGroup: false, decimalSeparator: 'comma', normalizeLPtoG: true,
      groupDimensions: !!request.grouped, mergeAcrossOrders: false,
    });
    result = {rows, preview};
  } else if (request.operation === 'processing_pdf') {
    const blob = await context.buildProcessingPdfBlob();
    result = {pdf_base64: Buffer.from(await blob.arrayBuffer()).toString('base64')};
  } else if (request.operation === 'labels_pdf') {
    const lines = request.preview.groups.flatMap(group => group.lines);
    const rows = lines.flatMap(line => context.buildProcessingLabelRows(line, line.orderId, line.client));
    result = {pdf_base64: Buffer.from(await context.generateLabelsPdf(rows)).toString('base64')};
  } else if (request.operation === 'invoice') {
    // Use the website's manual adapter, including its per-unit area conversion.
    const order = request.order.source === 'manual' ? context.manualOrderToShared(request.order, {perUnitArea: true}) : request.order;
    const job = context.buildInvoiceJobFromOrder(order);
    const lines = await context.buildInvoiceLinesFromRaw(context.ensureInvoiceRawRows(job), {allowAi: false, promptAllowedForJob: false});
    context.finalizeInvoiceTotals(job, lines);
    result = job;
  } else {
    throw new Error('Unsupported workflow');
  }
  process.stdout.write(JSON.stringify(result));
}
main().catch(() => { process.stderr.write('WORKFLOW_FAILED\n'); process.exitCode = 1; });
