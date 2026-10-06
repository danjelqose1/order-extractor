// Transport fixtures exercise the real editable prompt and PDF renderer without API billing.
const assert=require('node:assert/strict');
const path=require('node:path');
module.exports=async function dictationQA(page,name,output,aiBodies){
  let sessionGate=null;
  const sessions=[],closes=[],unexpected=[],downloads=[];
  page.on('download',item=>downloads.push(item));
  await page.route('**/api/production-sheets/voice/**',async route=>{
    const body=route.request().postDataJSON(),url=route.request().url();
    if(url.endsWith('/session')){
      sessions.push(body);if(sessionGate)await sessionGate;
      return route.fulfill({json:{session_id:'rtc_fixture',token:'t'.repeat(43),sdp:'v=0 fixture answer',model:'gpt-live-transcribe'}});
    }
    if(url.endsWith('/close')){closes.push(body);return route.fulfill({json:{ok:true}});}
    unexpected.push(body);return route.fulfill({status:410,json:{detail:'No spoken actions'}});
  });
  await page.evaluate(()=>{
    const now=Date.now;window.__dict={sent:[],stopped:0,offset:0,level:0,denied:false};
    Date.now=()=>now()+__dict.offset;
    const fixtureMicrophone=async()=>{
      if(__dict.denied)throw new DOMException('Denied','NotAllowedError');
      const track={enabled:true,kind:'audio',stop(){if(!this.stopped)__dict.stopped++;this.stopped=true;}};
      __dict.mic=track;return {getTracks:()=>[track],getAudioTracks:()=>[track]};
    };
    Object.defineProperty(Object.getPrototypeOf(navigator.mediaDevices),'getUserMedia',{configurable:true,value:fixtureMicrophone});
    window.AudioContext=class{
      resume(){return Promise.resolve();}close(){return Promise.resolve();}
      createMediaStreamSource(){return {connect(){}};}
      createAnalyser(){return {fftSize:1024,getFloatTimeDomainData(data){data.fill(__dict.level);}};}
    };
    window.RTCPeerConnection=class extends EventTarget{
      constructor(){super();this.iceGatheringState='complete';this.connectionState='connected';}
      addTrack(){}createOffer(){return Promise.resolve({type:'offer',sdp:'v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111'});}
      setLocalDescription(value){this.localDescription=value;return Promise.resolve();}
      setRemoteDescription(){
        this.channel.readyState='open';
        setTimeout(()=>__dict.emit({type:'session.created',session:{type:'transcription'},event_id:'start-'+Math.random()}),5);
        return Promise.resolve();
      }
      createDataChannel(label){
        __dict.channelLabel=label;const channel=new EventTarget();channel.readyState='connecting';
        channel.send=data=>__dict.sent.push(JSON.parse(data));
        this.channel=channel;__dict.emit=event=>channel.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(event)}));
        return channel;
      }
      close(){this.connectionState='closed';}
    };
    ProductionSheetVoice.update();
  });
  const original=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
  const input=page.locator('#productionSheetRequest');
  const idle=async()=>page.waitForFunction(()=>!document.getElementById('productionSheetVoiceStart').disabled);
  const start=async()=>{
    await page.locator('#productionSheetVoiceStart').click();
    await page.waitForFunction(()=>document.getElementById('productionSheetVoiceStatus').textContent.includes('Listening'));
    assert.equal(await page.evaluate(()=>__dict.channelLabel),'oai-events');
  };
  const emit=event=>page.evaluate(event=>__dict.emit(event),event);
  const completed=async(id,text,previous=null)=>{
    await emit({type:'input_audio_buffer.committed',item_id:id,previous_item_id:previous});
    await emit({type:'conversation.item.input_audio_transcription.completed',item_id:id,transcript:text,event_id:id+'-final'});
  };
  const count=aiBodies.length;
  await input.fill('Rrite shkrimin.');
  await start();
  await emit({type:'input_audio_buffer.committed',item_id:'sq',previous_item_id:null});
  const delta={type:'conversation.item.input_audio_transcription.delta',item_id:'sq',delta:'Vendos tri ',event_id:'sq-delta'};
  await emit(delta);await emit(delta);
  assert.equal(await input.inputValue(),'Rrite shkrimin. Vendos tri');
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'sq',delta:'kolona.',event_id:'sq-delta2'});
  await completed('it','Lascia spazio dopo il tipo di vetro.','sq');
  await emit({type:'conversation.item.input_audio_transcription.completed',item_id:'sq',transcript:'Vendos tri kolona.',event_id:'sq-final'});
  assert.equal(await input.inputValue(),'Rrite shkrimin. Vendos tri kolona. Lascia spazio dopo il tipo di vetro.');
  assert.equal(aiBodies.length,count,'dictation must not send to Sol automatically');
  assert.equal(downloads.length,0);
  assert(await page.locator('#productionSheetReview').isHidden());
  assert(!await page.locator('#productionSheetPrint').isDisabled());
  await page.screenshot({path:path.join(output,name+'-dictation-draft.png'),fullPage:true});
  await page.locator('#productionSheetVoiceEnd').click();await idle();
  await input.fill('Rrite shkrimin. Vendos tri kolona. Shto hapësirë pas llojit të xhamit.');
  const text=await input.inputValue();
  await page.locator('#productionSheetAsk').click();
  await page.waitForFunction(()=>!document.getElementById('productionSheetApply').disabled);
  assert.equal(aiBodies.length,count+1);
  assert.equal(aiBodies.at(-1).instruction,text);
  assert(aiBodies.at(-1).images[0].startsWith('data:image/jpeg;base64,'));
  assert(await page.locator('#productionSheetPrint').isDisabled());
  assert(await page.locator('#productionSheetVoiceStart').isDisabled(),'apply/discard remains an explicit button action');
  await page.locator('#productionSheetApply').click();
  const downloadEvent=page.waitForEvent('download');
  await page.locator('#productionSheetSave').click();await downloadEvent;
  await downloads[0].saveAs(path.join(output,name+'-dictation.pdf'));
  assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),original);

  // A Send click stops capture and waits for the final text; no extra intent-routing call.
  await input.fill('');await start();
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'flush',delta:'Utilizza due',event_id:'flush-delta'});
  const beforeFlush=aiBodies.length;
  await page.locator('#productionSheetAsk').click();
  await page.waitForFunction(()=>__dict.sent.some(event=>event.type==='input_audio_buffer.commit'));
  assert.equal(await page.evaluate(()=>__dict.mic.enabled),false);
  await page.waitForTimeout(120);assert.equal(aiBodies.length,beforeFlush);
  await completed('flush','Utilizza due colonne.');
  await page.waitForFunction(()=>!document.getElementById('productionSheetApply').disabled);
  assert.equal(aiBodies.length,beforeFlush+1);
  assert.equal(aiBodies.at(-1).instruction,'Utilizza due colonne.');
  await page.locator('#productionSheetDiscard').click();
  await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
  assert.equal(unexpected.length,0);
  assert(!(await page.evaluate(()=>__dict.sent)).some(event=>/response.create|session.commentary|delegation/.test(event.type)));

  // A handwritten correction wins over late transcription.
  await input.fill('');await start();
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'edit',delta:'Wrong draft',event_id:'edit-delta'});
  await input.fill('My corrected request');await idle();
  await emit({type:'conversation.item.input_audio_transcription.completed',item_id:'edit',transcript:'Late overwrite'});
  assert.equal(await input.inputValue(),'My corrected request');

  // A failed finalization keeps the partial draft, requires review, and does not send.
  await input.fill('');await start();
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'timeout',delta:'Bëj tre kolona',event_id:'timeout-delta'});
  const beforeTimeout=aiBodies.length;
  await page.locator('#productionSheetAsk').click();await idle();
  assert.equal(aiBodies.length,beforeTimeout);
  assert.equal(await input.inputValue(),'Bëj tre kolona');
  assert.match(await page.locator('#productionSheetStatus').innerText(),/Check the dictated text/);

  // Silence disconnects; typing/scrolling cannot keep paid transcription alive.
  await input.fill('');await start();
  await page.evaluate(()=>{__dict.offset+=61000;});await idle();
  assert.match(await page.locator('#productionSheetVoiceStatus').innerText(),/1 minute/);
  assert(closes.length>=1 && await page.evaluate(()=>__dict.stopped>0));

  // Cancel creation and hang up the late negotiated call.
  let release;sessionGate=new Promise(resolve=>{release=resolve;});
  const creating=page.waitForRequest(request=>request.url().endsWith('/voice/session'));
  await page.locator('#productionSheetVoiceStart').click();await creating;
  await page.locator('#productionSheetVoiceEnd').click();await idle();
  const closeCount=closes.length;release();sessionGate=null;
  await page.waitForTimeout(100);assert.equal(closes.length,closeCount+1);

  await start();
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'failure',delta:'Rrite shkrimin',event_id:'failure-delta'});
  await emit({type:'error',error:{message:'transport failure'}});await idle();
  assert.equal(await input.inputValue(),'Rrite shkrimin');
  assert(!await page.locator('#productionSheetPrint').isDisabled());

  await start();
  await page.evaluate(()=>{appState.processing.rows[0].width+=5;recalcProcessingPreview();updateProcessingUI();});
  await page.waitForFunction(()=>document.getElementById('productionSheetVoiceEnd').hidden);
  assert(await page.locator('#productionSheetVoiceStart').isDisabled());
  await page.locator('#productionSheetRefresh').click();
  await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
  await page.evaluate(()=>{__dict.denied=true;});await page.locator('#productionSheetVoiceStart').click();await idle();
  assert.match(await page.locator('#productionSheetVoiceStatus').innerText(),/Allow microphone/);
  await page.evaluate(()=>{__dict.denied=false;});await start();
  await emit({type:'conversation.item.input_audio_transcription.delta',item_id:'long',delta:'ë'.repeat(2100)});
  await idle();assert.equal((await input.inputValue()).length,2000);
  assert.match(await page.locator('#productionSheetVoiceStatus').innerText(),/limit/);
  await input.fill('');await start();
  await page.locator('#productionSheetClose').click();
  await page.waitForFunction(()=>document.getElementById('productionSheetVoiceEnd').hidden);
  assert.equal(await page.evaluate(()=>__dict.mic.stopped),true);
  assert(sessions.every(session=>Object.keys(session).join(',')==='sdp'),'no production data is sent to the transcription service');
  console.log(name+': multilingual editable dictation, manual Send only, ordered/duplicate transcripts, final-text flush, corrections, timeout, PDF review/save, idle cleanup, cancellation, source changes and microphone denial passed');
};
