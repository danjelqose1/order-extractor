/* Dictation fills an editable prompt. Only the user's Send click can ask the PDF formatter. */
(function(root){
  "use strict";
  const el=id=>document.getElementById(id), dialog=el("productionSheetDialog"), request=el("productionSheetRequest");
  if (!dialog || !el("productionSheetVoiceStart") || !request) return;
  const idleMs=60000, maxSessionMs=15*60000, silenceMs=900, finalizeMs=2500;
  let connection=null;
  const supported=()=>!!(root.RTCPeerConnection && navigator.mediaDevices?.getUserMedia && (root.AudioContext || root.webkitAudioContext));
  const status=text=>{el("productionSheetVoiceStatus").textContent=text;};
  function context(){return root.ProductionSheetUI.dictationContext();}
  function update(){
    let ready=false;try{context();ready=true;}catch{}
    el("productionSheetVoiceStart").disabled=!!connection || !ready || !supported();
    el("productionSheetVoiceStart").hidden=!!connection;
    el("productionSheetVoiceEnd").hidden=!connection;
    el("productionSheetVoiceEnd").disabled=!!connection?.closing;
    if(!supported())status("Ky shfletues nuk mbështet mikrofonin. / Microphone support is required.");
  }
  function send(c,type){
    if(c.channel?.readyState!=="open")return false;
    c.channel.send(JSON.stringify({type,event_id:"dictation_"+crypto.randomUUID()}));return true;
  }
  async function post(body){
    const response=await fetch(API_BASE+"/api/production-sheets/voice/session",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const result=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error("Diktimi nuk u lidh. Mund ta shkruani kërkesën. / Dictation did not connect. Type your request instead.");
    return result;
  }
  function hangup(c,keepalive=false){
    if(!c.owner || c.hangupSent)return;c.hangupSent=true;
    fetch(API_BASE+"/api/production-sheets/voice/close",{method:"POST",keepalive,
      headers:{"Content-Type":"application/json"},body:JSON.stringify(c.owner)}).catch(()=>{});
  }
  function release(c,complete=true,unloading=false){
    if(c.released)return;c.released=true;
    if(connection===c)connection=null;
    clearInterval(c.timer);clearTimeout(c.closeTimer);clearTimeout(c.startTimer);
    c.stream?.getTracks().forEach(track=>track.stop());c.peer?.close();
    c.audioContext?.close().catch(()=>{});hangup(c,unloading);
    update();
    c.resolveDone(complete);
  }
  function commit(c){
    if(!c.uncommitted)return;
    c.uncommitted=false;c.voicedMs=0;
    if(send(c,"input_audio_buffer.commit"))c.outstanding++;
  }
  function end(message="Diktimi përfundoi. Kontrolloni tekstin dhe shtypni Dërgo. / Dictation stopped. Review the text, then tap Send.",immediate=false,unloading=false){
    const c=connection;if(!c)return Promise.resolve(true);
    if(c.closing){if(immediate)release(c,false,unloading);return c.done;}
    c.closing=true;status(message);clearInterval(c.timer);clearTimeout(c.startTimer);
    c.stream?.getTracks().forEach(track=>{track.enabled=false;});update();
    if(immediate || !c.started){release(c,!c.started,unloading);return c.done;}
    commit(c);
    if(!c.outstanding && [...c.items.values()].every(item=>item.final)){release(c);return c.done;}
    c.closeTimer=setTimeout(()=>{
      status("Kontrolloni tekstin para se të dërgoni. / Final transcript unavailable. Review the text before sending.");
      release(c,false);
    },finalizeMs);
    return c.done;
  }
  function item(c,id){
    if(typeof id!=="string" || !id || id.length>200)return null;
    if(!c.items.has(id)){c.items.set(id,{id,text:"",final:false});c.order.push(id);}
    return c.items.get(id);
  }
  function publish(c){
    if(c.freeze || connection!==c)return;
    const words=c.order.map(id=>c.items.get(id).text.trim()).filter(Boolean).join(" ");
    const text=c.prefix+(c.prefix && words && !/\s$/.test(c.prefix)?" ":"")+words;
    request.value=text.slice(0,request.maxLength);
    if(text.length>request.maxLength)void end("Kërkesa arriti kufirin. Kontrollojeni para se ta dërgoni. / Request limit reached. Review before sending.",true);
  }
  function onEvent(c,raw){
    if(connection!==c || typeof raw!=="string" || raw.length>100000)return;
    let event;try{event=JSON.parse(raw);}catch{return;}
    if(event.event_id){
      if(c.events.has(event.event_id))return;c.events.add(event.event_id);
      if(c.events.size>1000)c.events.delete(c.events.values().next().value);
    }
    if(["session.created","session.updated","transcription_session.created","transcription_session.updated"].includes(event.type)){
      if(c.closing)return;
      if(event.session?.type && event.session.type!=="transcription"){
        void end("Rifreskoni faqen për diktimin e ri. / Reload the page for dictation.",true);return;
      }
      c.started=true;c.lastActivity=Date.now();clearTimeout(c.startTimer);
      status("Po dëgjoj… Teksti shfaqet më poshtë. / Listening… Your request appears below.");update();return;
    }
    if(event.type==="input_audio_buffer.committed"){
      const current=item(c,event.item_id);if(!current)return;
      current.committed=true;
      // Commit events establish speech order; final transcripts can arrive out of order.
      c.order=c.order.filter(id=>id!==event.item_id);
      const previous=c.order.indexOf(event.previous_item_id);
      c.order.splice(event.previous_item_id===null?0:previous<0?c.order.length:previous+1,0,event.item_id);
      publish(c);return;
    }
    if(event.type==="conversation.item.input_audio_transcription.delta" || event.type==="conversation.item.input_audio_transcription.completed"){
      const current=item(c,event.item_id);if(!current || current.final || c.freeze)return;
      if(event.type.endsWith(".delta")){
        if(typeof event.delta!=="string")return;current.text+=event.delta;
        if(!current.committed)c.uncommitted=true;
      }else{
        if(typeof event.transcript!=="string")return;current.text=event.transcript;current.final=true;
        c.outstanding=Math.max(0,c.outstanding-1);
      }
      c.lastActivity=Date.now();publish(c);
      if(c.closing && !c.outstanding && [...c.items.values()].every(part=>part.final))release(c);
      return;
    }
    if(event.type==="error" || event.type==="conversation.item.input_audio_transcription.failed"){
      void end("Diktimi u ndërpre. Kontrolloni ose shkruani kërkesën. / Dictation interrupted. Review or type your request.",true);
    }
  }
  function meter(c,stream){
    const source=c.audioContext.createMediaStreamSource(stream),analyser=c.audioContext.createAnalyser();
    analyser.fftSize=1024;source.connect(analyser);
    return {source,analyser,data:new Float32Array(analyser.fftSize),floor:.01};
  }
  function tick(c){
    if(connection!==c || c.closing || !c.started)return;
    const now=Date.now(),input=c.input;
    input.analyser.getFloatTimeDomainData(input.data);
    const level=Math.sqrt(input.data.reduce((sum,value)=>sum+value*value,0)/input.data.length);
    if(level>Math.max(.018,input.floor*2.8)){
      c.lastSpeech=now;c.lastActivity=now;c.voicedMs+=200;c.uncommitted=true;
    }else{
      input.floor=input.floor*.98+Math.min(level,input.floor*1.5)*.02;
      if(c.uncommitted && c.voicedMs>=200 && now-c.lastSpeech>=silenceMs)commit(c);
    }
    if(now-c.created>=maxSessionMs)void end("Diktimi përfundoi pas 15 minutash. / Dictation stopped after 15 minutes.");
    else if(now-c.lastActivity>=idleMs)void end("U shkëput pas 1 minute pa të folur. / Disconnected after 1 minute of inactivity.");
  }
  async function start(){
    if(connection)return;
    let sheet;try{sheet=context();}catch(error){status(error.message);return;}
    if(!supported()){update();return;}
    if(request.value.length>=request.maxLength){status("Shkurtoni kërkesën para diktimit. / Shorten the request before dictating.");return;}
    const c={created:Date.now(),lastActivity:Date.now(),lastSpeech:0,voicedMs:0,uncommitted:false,outstanding:0,
      prefix:request.value,items:new Map(),order:[],events:new Set(),started:false,closing:false,freeze:false};
    c.done=new Promise(resolve=>{c.resolveDone=resolve;});
    connection=c;status("Po lidh mikrofonin… / Connecting microphone…");update();
    try{
      c.audioContext=new (root.AudioContext || root.webkitAudioContext)();await c.audioContext.resume();
      c.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      if(connection!==c || c.closing){c.stream.getTracks().forEach(track=>track.stop());return;}
      c.input=meter(c,c.stream);c.peer=new RTCPeerConnection();
      c.stream.getTracks().forEach(track=>c.peer.addTrack(track,c.stream));
      c.peer.addEventListener("connectionstatechange",()=>{
        if(connection===c && !c.closing && ["failed","disconnected","closed"].includes(c.peer.connectionState))
          void end("Lidhja u ndërpre. Teksti mbetet i shkruar. / Connection lost. Your draft is kept.",true);
      });
      c.channel=c.peer.createDataChannel("oai-events");
      c.channel.addEventListener("message",event=>onEvent(c,event.data));
      c.channel.addEventListener("close",()=>{
        if(connection===c)void end("Diktimi u mbyll. Teksti mbetet i shkruar. / Dictation disconnected. Your draft is kept.",true);
      });
      await c.peer.setLocalDescription(await c.peer.createOffer());
      await new Promise((resolve,reject)=>{
        if(c.peer.iceGatheringState==="complete")return resolve();
        const timer=setTimeout(()=>{c.peer.removeEventListener("icegatheringstatechange",changed);reject(new Error("Microphone connection timed out."));},10000);
        function changed(){if(c.peer.iceGatheringState==="complete"){clearTimeout(timer);c.peer.removeEventListener("icegatheringstatechange",changed);resolve();}}
        c.peer.addEventListener("icegatheringstatechange",changed);
      });
      if(connection!==c || c.closing)return;
      // A late creation answer must still be hung up after cancellation.
      const result=await post({sdp:c.peer.localDescription.sdp});
      c.owner={session_id:result.session_id,token:result.token};
      if(connection!==c || c.closing){hangup(c);return;}
      if(result.model!=="gpt-live-transcribe")throw new Error("Reload the page for the new dictation mode.");
      if(context().source_digest!==sheet.source_digest)throw new Error("The sheet changed. Start dictation again.");
      await c.peer.setRemoteDescription({type:"answer",sdp:result.sdp});
      if(connection!==c || c.closing)return;
      c.timer=setInterval(()=>tick(c),200);
      c.startTimer=setTimeout(()=>{if(!c.started && connection===c)void end("Diktimi nuk u lidh. Shkruani kërkesën ose provoni sërish. / Dictation did not connect. Type or try again.",true);},15000);
    }catch(error){
      if(connection===c && !c.closing)void end(error.name==="NotAllowedError"
        ?"Lejoni mikrofonin në shfletues. / Allow microphone access, then try again.":error.message,true);
      if(connection!==c || c.closing)c.stream?.getTracks().forEach(track=>track.stop());
    }
  }
  root.ProductionSheetVoice={update,end,finish:()=>end(),active:()=>!!connection};
  el("productionSheetVoiceStart").addEventListener("click",()=>void start());
  el("productionSheetVoiceEnd").addEventListener("click",()=>void end());
  request.addEventListener("input",()=>{
    if(!connection)return;connection.freeze=true;
    void end("Teksti u ndryshua. Diktimi u ndal. / Text edited. Dictation stopped.",true);
  });
  root.addEventListener("pagehide",()=>void end("",true,true));
  update();
})(window);
