// Transport fixtures exercise the real sheet UI and renderer without audio/API billing.
const assert=require('node:assert/strict');
const path=require('node:path');
module.exports=async function voiceQA(page,name,output,aiBodies){
  let decision={action:'clarify',instruction:'',reply:'Si mund t’ju ndihmoj?'}, turnGate=null, sessionGate=null;
  const turns=[],sessions=[],closes=[],downloads=[];
  page.on('download',item=>downloads.push(item));
  await page.route('**/api/production-sheets/voice/**',async route=>{
    const body=route.request().postDataJSON(),url=route.request().url();
    if(url.endsWith('/session')){
      sessions.push(body);if(sessionGate)await sessionGate;
      return route.fulfill({json:{session_id:'live_fixture',token:'t'.repeat(43),sdp:'v=0 fixture answer'}});
    }
    if(url.endsWith('/close')){closes.push(body);return route.fulfill({json:{ok:true}});}
    turns.push(body);const result={...decision};if(turnGate)await turnGate;
    return route.fulfill({json:result});
  });
  await page.evaluate(()=>{
    const now=Date.now;window.__voice={sent:[],stopped:0,offset:0,levels:{input:0,output:0},denied:false,noClosed:false};
    Date.now=()=>now()+__voice.offset;
    const track=()=>({enabled:true,kind:'audio',stop(){__voice.stopped++;}});
    const fixtureMicrophone=async()=>{
      if(__voice.denied)throw new DOMException('Denied','NotAllowedError');
      const t=track();return {role:'input',getTracks:()=>[t],getAudioTracks:()=>[t]};
    };
    // WebKit may recreate the MediaDevices wrapper after collection; patch its
    // prototype so reconnects retain the fixture rather than opening a real mic.
    Object.defineProperty(Object.getPrototypeOf(navigator.mediaDevices),'getUserMedia',{configurable:true,value:fixtureMicrophone});
    window.MediaStream=class{constructor(){this.role='output';}};
    window.AudioContext=class{
      resume(){return Promise.resolve();}close(){return Promise.resolve();}
      createMediaStreamSource(stream){return {connect(analyser){analyser.role=stream.role;}};}
      createAnalyser(){return {fftSize:1024,getFloatTimeDomainData(data){data.fill(__voice.levels[this.role]||0);}};}
    };
    window.RTCPeerConnection=class extends EventTarget{
      constructor(){super();this.iceGatheringState='complete';this.connectionState='connected';}
      addTrack(){}createOffer(){return Promise.resolve({type:'offer',sdp:'v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111'});}
      setLocalDescription(value){this.localDescription=value;return Promise.resolve();}
      setRemoteDescription(){
        this.channel.readyState='open';
        setTimeout(()=>__voice.emit({type:'session.started',event_id:'start-'+Math.random()}),5);
        return Promise.resolve();
      }
      createDataChannel(label){
        __voice.channelLabel=label;const channel=new EventTarget();channel.readyState='connecting';
        channel.send=data=>{
          const event=JSON.parse(data);__voice.sent.push(event);
          if(event.type==='session.close' && !__voice.noClosed)setTimeout(()=>__voice.emit({type:'session.closed',usage:{seconds:15}}),5);
        };
        this.channel=channel;__voice.emit=event=>channel.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(event)}));
        __voice.track=()=>{const event=new Event('track');event.track=track();this.dispatchEvent(event);};
        return channel;
      }
      close(){this.connectionState='closed';}
    };
    // Suppress fixture autoplay failure; real autoplay has a separate Play control.
    HTMLMediaElement.prototype.play=()=>Promise.resolve();
    Object.defineProperty(HTMLMediaElement.prototype,'srcObject',{configurable:true,get(){return this._fixtureStream;},set(value){this._fixtureStream=value;}});
    ProductionSheetVoice.update();
  });
  const original=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
  const start=async()=>{
    await page.locator('#productionSheetVoiceStart').click();
    try{await page.waitForFunction(()=>!document.getElementById('productionSheetVoiceMute').hidden);}catch(error){
      console.error(await page.evaluate(()=>({status:document.getElementById('productionSheetVoiceStatus').textContent,sent:__voice.sent.slice(-4),disabled:document.getElementById('productionSheetVoiceStart').disabled})));
      throw error;
    }
    assert.equal(await page.evaluate(()=>__voice.channelLabel),'oai-events');
  };
  const spoken=async(text,id)=>page.evaluate(({text,id})=>{
    __voice.emit({type:'session.input_transcript.delta',delta:text,event_id:id+'-text'});
    __voice.emit({type:'session.delegation.created',event_id:id+'-event',delegation:{id,type:'delegation',target:'client'}});
  },{text,id});
  const result=async id=>page.waitForFunction(id=>__voice.sent.some(e=>e.type==='session.commentary.append' && e.delegation_id===id),id);
  const idle=async()=>page.waitForFunction(()=>!document.getElementById('productionSheetVoiceStart').disabled);
  await start();
  assert((await page.evaluate(()=>__voice.sent)).some(e=>e.type==='session.instructions.append' && /Albanian/.test(e.content)));
  decision={action:'propose',instruction:'Rrite shkrimin dhe shto hapësirë pas llojit të xhamit.',reply:'Po.'};
  await spoken(decision.instruction,'format-sq');await result('format-sq');
  assert.equal(aiBodies.at(-1).instruction,decision.instruction);
  assert(aiBodies.at(-1).images[0].startsWith('data:image/jpeg;base64,'));
  assert(await page.locator('#productionSheetProposal').isVisible());
  assert(await page.locator('#productionSheetPrint').isDisabled());
  assert.match(turns.at(-1).transcript,/USER: Rrite/);
  await page.locator('#productionSheetVoiceTranscript').evaluate(e=>e.parentElement.open=true);
  await page.screenshot({path:path.join(output,`${name}-voice-proposal.png`),fullPage:true});
  decision={action:'apply',instruction:'',reply:'Po.'};
  await spoken('Po, aplikoje këtë propozim.','apply-sq');await result('apply-sq');
  assert(!await page.locator('#productionSheetPrint').isDisabled());
  const count=turns.length;
  await page.evaluate(()=>__voice.emit({type:'session.delegation.created',delegation:{id:'apply-sq',target:'client'}}));
  await page.waitForTimeout(80);assert.equal(turns.length,count,'duplicate delegation must not repeat actions');
  decision={action:'save_pdf',instruction:'',reply:'Po.'};
  await spoken('Salva il PDF, per favore.','save-it');await result('save-it');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'productionSheetSave');
  assert.equal(downloads.length,0);
  const downloadEvent=page.waitForEvent('download');
  await page.locator('#productionSheetSave').click();await downloadEvent;
  assert.equal(downloads.length,1);await downloads[0].saveAs(path.join(output,`${name}-voice.pdf`));
  assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),original);
  await page.locator('#productionSheetVoiceMute').click();
  assert.equal(await page.locator('#productionSheetVoiceMute').getAttribute('aria-pressed'),'true');
  assert((await page.evaluate(()=>__voice.sent)).some(e=>e.type==='session.input_audio.mute'));
  await page.locator('#productionSheetVoiceMute').click();
  await page.evaluate(()=>{__voice.track();__voice.levels.output=.03;__voice.offset+=70000;});
  await page.waitForTimeout(300);
  assert(await page.locator('#productionSheetVoiceStart').isDisabled(),'playback pauses inactivity');
  await page.evaluate(()=>{__voice.levels.output=0;__voice.offset+=61000;});await idle();
  assert.match(await page.locator('#productionSheetVoiceStatus').innerText(),/1 minute/);
  assert(closes.length>=1 && await page.evaluate(()=>__voice.stopped>0));

  // Pending routing and the slower visual formatter pause idle; no stale approval.
  await start();let release;
  turnGate=new Promise(resolve=>{release=resolve;});decision={action:'apply',instruction:'',reply:'Po.'};
  const requested=page.waitForRequest(r=>r.url().endsWith('/voice/turn'));
  await spoken('Po','pending');await requested;
  await page.evaluate(()=>{__voice.offset+=70000;});await page.waitForTimeout(250);
  assert(await page.locator('#productionSheetVoiceStart').isDisabled(),'backend work pauses inactivity');
  await page.locator('#productionSheetVoiceEnd').click();await idle();release();turnGate=null;
  await page.waitForTimeout(100);
  assert(!await page.evaluate(()=>__voice.sent.some(e=>e.delegation_id==='pending' && e.type==='session.commentary.append')));

  // Cancel during session creation: microphone stops and the late session is hung up.
  sessionGate=new Promise(resolve=>{release=resolve;});
  const creating=page.waitForRequest(r=>r.url().endsWith('/voice/session'));
  await page.locator('#productionSheetVoiceStart').click();
  try{await creating;}catch(error){
    console.error(await page.evaluate(()=>({status:document.getElementById('productionSheetVoiceStatus').textContent,sent:__voice.sent.slice(-4),disabled:document.getElementById('productionSheetVoiceStart').disabled})));
    throw error;
  }
  await page.locator('#productionSheetVoiceEnd').click();await idle();
  const closeCount=closes.length;release();sessionGate=null;
  await page.waitForFunction(()=>!document.getElementById('productionSheetVoiceStart').disabled);
  await page.waitForTimeout(100);assert.equal(closes.length,closeCount+1);

  // Manual changes while a routed request is pending invalidate its result.
  await start();turnGate=new Promise(resolve=>{release=resolve;});
  decision={action:'save_pdf',instruction:'',reply:'Po.'};
  const changedRequest=page.waitForRequest(r=>r.url().endsWith('/voice/turn'));
  await spoken('Save PDF','changed');await changedRequest;
  await page.locator('#productionSheetReset').click();
  await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
  release();turnGate=null;await result('changed');assert.equal(downloads.length,1);
  assert.match(await page.evaluate(()=>__voice.sent.find(e=>e.delegation_id==='changed').content),/sheet changed/i);
  await page.evaluate(()=>{appState.processing.rows[0].width+=5;recalcProcessingPreview();updateProcessingUI();});
  await page.waitForFunction(()=>document.getElementById('productionSheetVoiceEnd').hidden);
  assert(await page.locator('#productionSheetVoiceStart').isDisabled());
  await page.locator('#productionSheetRefresh').click();
  await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
  await page.evaluate(()=>{__voice.denied=true;});await page.locator('#productionSheetVoiceStart').click();await idle();
  assert.match(await page.locator('#productionSheetVoiceStatus').innerText(),/Allow microphone/);
  await page.evaluate(()=>{__voice.denied=false;});await start();
  await page.locator('#productionSheetClose').click();await page.waitForFunction(()=>document.getElementById('productionSheetVoiceEnd').hidden);
  assert(sessions.every(s=>s.context.source_digest.length===64));
  console.log(`${name}: multilingual voice proposals/apply/PDF, duplicate events, playback/work-aware idle, late connection cleanup, cancellation, source changes and microphone denial passed`);
};
