/* GPT-Live carries speech; the existing immutable PDF workflow owns every sheet action. */
(function(root){
  "use strict";
  const el=id=>document.getElementById(id), dialog=el("productionSheetDialog");
  if (!dialog || !el("productionSheetVoiceStart")) return;
  const idleMs=60000, maxSessionMs=15*60000;
  let connection=null;
  const supported=()=>!!(root.RTCPeerConnection && navigator.mediaDevices?.getUserMedia && (root.AudioContext || root.webkitAudioContext));
  function status(text){el("productionSheetVoiceStatus").textContent=text;}
  function context(){return root.ProductionSheetUI.voiceContext();}
  function update(){
    let ready=false;try{context();ready=true;}catch{}
    el("productionSheetVoiceStart").disabled=!!connection || !ready || !supported();
    el("productionSheetVoiceEnd").hidden=!connection;
    el("productionSheetVoiceMute").hidden=!connection?.started || connection.closing;
    if(!supported()) status("Ky shfletues nuk mbështet zërin. / Voice needs a browser with microphone support.");
    if(connection?.started && !connection.closing && ready){
      const text=JSON.stringify(context());
      if(text!==connection.contextText){
        connection.contextText=text;
        send(connection,"session.thinking.append",{delegation_id:null,content:"Current read-only sheet state: "+text.slice(0,1000)});
      }
    }
  }
  function send(c,type,fields={}){
    if(c.channel?.readyState!=="open") return false;
    c.channel.send(JSON.stringify({type,event_id:`sheet_${crypto.randomUUID()}`,...fields}));return true;
  }
  async function post(path,body,signal){
    const response=await fetch(`${API_BASE}/api/production-sheets/voice/${path}`,{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal});
    const result=await response.json().catch(()=>({}));
    if(!response.ok) throw new Error(typeof result.detail==="string"?result.detail:"Voice is unavailable. Your sheet is unchanged.");
    return result;
  }
  function hangup(c,keepalive=false){
    if(!c.owner || c.hangupSent) return;
    c.hangupSent=true;
    // Ownership token is scoped to this conversation; the OpenAI key never reaches the browser.
    fetch(`${API_BASE}/api/production-sheets/voice/close`,{method:"POST",keepalive,
      headers:{"Content-Type":"application/json"},body:JSON.stringify(c.owner)}).catch(()=>{});
  }
  function release(c){
    clearInterval(c.timer);clearTimeout(c.closeTimer);clearTimeout(c.startTimer);
    c.controller?.abort();c.stream?.getTracks().forEach(track=>track.stop());
    c.peer?.close();c.audio?.pause();if(c.audio)c.audio.srcObject=null;
    c.audioContext?.close().catch(()=>{});
    if(connection===c){connection=null;el("productionSheetVoicePlay").hidden=true;update();}
  }
  function end(message="Biseda përfundoi. / Conversation ended.",unloading=false){
    const c=connection;if(!c || c.closing)return;
    c.closing=true;status(message);
    c.controller?.abort();c.stream?.getTracks().forEach(track=>{track.enabled=false;});c.audio?.pause();
    clearInterval(c.timer);clearTimeout(c.startTimer);update();
    const sent=send(c,"session.close");
    // Silence input/output now, retain the negotiated transport until final usage.
    // A server hangup is a fallback, since racing it with close can lose session.closed.
    if(unloading || !sent){hangup(c,unloading);release(c);}
    else c.closeTimer=setTimeout(()=>{hangup(c);release(c);},10000);
  }
  function record(c,role,delta){
    if(typeof delta!=="string" || !delta)return;
    c.lastActivity=Date.now();if(role==="user")c.inputVersion++;
    const last=c.messages.at(-1);
    if(last?.role===role)last.text+=delta;else c.messages.push({role,text:delta});
    // Full-duplex fragments remain separate and preserve their original spaces.
    while(c.messages.length>60)c.messages.shift();
    for(const item of c.messages)item.text=item.text.slice(-8000);
    el("productionSheetVoiceTranscript").textContent=c.messages.slice(-8)
      .map(item=>`${item.role==="user"?"Ju / You":"AI"}: ${item.text}`).join("\n");
    el("productionSheetVoiceTranscript").scrollTop=el("productionSheetVoiceTranscript").scrollHeight;
  }
  async function delegate(c,event){
    const id=event.delegation?.id;
    if(!id || event.delegation.target!=="client" || c.delegations.has(id) || c.closing)return;
    c.delegations.add(id);
    if(c.working){send(c,"session.commentary.append",{delegation_id:id,content:"A sheet review is already running. Wait for its proposal before another action."});return;}
    c.working=true;c.lastActivity=Date.now();status("AI po kontrollon kërkesën dhe fletën… / Reviewing your request and sheet…");
    c.controller=new AbortController();const controller=c.controller;
    const timeout=setTimeout(()=>controller.abort(),100000);
    try{
      const expected=context(), version=c.inputVersion;
      const transcript=c.messages.map(m=>`${m.role.toUpperCase()}: ${m.text}`).join("\n").slice(-16000);
      if(!transcript)throw new Error("Nuk e dëgjova kërkesën. / Please repeat your request.");
      const decision=await post("turn",{...c.owner,transcript,context:expected},controller.signal);
      if(connection!==c || c.closing)return;
      if(version!==c.inputVersion)throw new Error("Dëgjova diçka tjetër. Përsëriteni kërkesën. / Please repeat your latest request.");
      const result=await root.ProductionSheetUI.voiceAction(decision,expected);
      if(connection!==c || c.closing)return;
      send(c,"session.commentary.append",{delegation_id:id,content:("Verified application result: "+result).slice(0,1000)});
    }catch(error){
      if(connection===c && !c.closing){
        const text=error.name==="AbortError"?"Kërkesa zgjati shumë. Provoni përsëri. / Please try again.":error.message;
        status(text);send(c,"session.commentary.append",{delegation_id:id,content:("The requested action did not complete. Explain in the user's language: "+text).slice(0,1000)});
      }
    }finally{
      clearTimeout(timeout);c.working=false;c.lastActivity=Date.now();
      if(c.controller===controller)c.controller=null;
      if(connection===c && !c.closing){status("Po dëgjoj… / Listening…");update();}
    }
  }
  function onEvent(c,raw){
    if(connection!==c || typeof raw!=="string" || raw.length>100000)return;
    let event;try{event=JSON.parse(raw);}catch{return;}
    if(event.event_id){if(c.events.has(event.event_id))return;c.events.add(event.event_id);if(c.events.size>1000)c.events.delete(c.events.values().next().value);}
    if(event.type==="session.closed"){
      c.usage=event.usage?.seconds;hangup(c);release(c);
      if(!c.closing)status("Biseda përfundoi. / Conversation ended.");return;
    }
    if(c.closing)return;
    switch(event.type){
      case "session.started":
        c.started=true;c.lastActivity=Date.now();clearTimeout(c.startTimer);
        status("Po dëgjoj… Flisni shqip ose në një gjuhë tjetër. / Listening… Speak in your language.");
        send(c,"session.instructions.append",{delegation_id:null,content:"Greet the user now briefly in Albanian: you can help format this production sheet. Then listen; follow the language the user speaks."});update();break;
      case "session.input_transcript.delta":record(c,"user",event.delta);break;
      case "session.output_transcript.delta":record(c,"assistant",event.delta);break;
      case "session.delegation.created":void delegate(c,event);break;
      case "session.usage.updated":c.usage=event.usage?.seconds;break;
      case "error":case "session.error":end("Zëri nuk është i disponueshëm. / Voice is unavailable. Your sheet is unchanged.");break;
    }
  }
  function meter(c,stream){
    const source=c.audioContext.createMediaStreamSource(stream),analyser=c.audioContext.createAnalyser();
    analyser.fftSize=1024;source.connect(analyser);
    return {source,analyser,data:new Float32Array(analyser.fftSize),floor:0.01};
  }
  function energy(m){
    if(!m)return 0;m.analyser.getFloatTimeDomainData(m.data);
    return Math.sqrt(m.data.reduce((sum,value)=>sum+value*value,0)/m.data.length);
  }
  function tick(c){
    if(connection!==c || c.closing || !c.started)return;
    const now=Date.now(),output=energy(c.output),input=c.muted?0:energy(c.input);
    // Learn steady background noise; speech rises above it. Playback is measured,
    // since transcript gaps alone do not prove the assistant has finished speaking.
    c.input.floor=c.input.floor*.98+Math.min(input,c.input.floor*1.5)*.02;
    if(input>Math.max(.018,c.input.floor*2.8) || output>.004 || c.working)c.lastActivity=now;
    if(now-c.created>maxSessionMs && !c.working && output<=.004)end("Biseda përfundoi pas 15 minutash. Mund ta nisni sërish. / Start again to continue.");
    else if(now-c.lastActivity>=idleMs)end("U shkëput pas 1 minute pa aktivitet. / Disconnected after 1 minute of inactivity.");
  }
  async function start(){
    if(connection)return;
    let sheet;try{sheet=context();}catch(error){status(error.message);return;}
    if(!supported()){update();return;}
    const c={created:Date.now(),lastActivity:Date.now(),messages:[],events:new Set(),delegations:new Set(),
      started:false,closing:false,working:false,muted:false,inputVersion:0,contextText:JSON.stringify(sheet)};
    connection=c;el("productionSheetVoiceTranscript").textContent="";status("Po lidh mikrofonin… / Connecting microphone…");update();
    el("productionSheetVoiceMute").textContent="Hesht mikrofonin / Mute";
    el("productionSheetVoiceMute").setAttribute("aria-pressed","false");
    try{
      c.audioContext=new (root.AudioContext || root.webkitAudioContext)();await c.audioContext.resume();
      c.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      if(connection!==c || c.closing){c.stream.getTracks().forEach(t=>t.stop());return;}
      c.input=meter(c,c.stream);c.audio=document.createElement("audio");c.audio.autoplay=true;
      c.peer=new RTCPeerConnection();c.stream.getTracks().forEach(track=>c.peer.addTrack(track,c.stream));
      c.peer.addEventListener("track",event=>{
        if(connection!==c || c.closing)return;
        const stream=new MediaStream([event.track]);c.output=meter(c,stream);c.audio.srcObject=stream;
        c.audio.play().catch(()=>{if(connection===c && !c.closing)el("productionSheetVoicePlay").hidden=false;});
      });
      c.peer.addEventListener("connectionstatechange",()=>{
        if(connection===c && !c.closing && ["failed","disconnected","closed"].includes(c.peer.connectionState))end("Lidhja u ndërpre. / Connection lost. Start voice again.");
      });
      c.channel=c.peer.createDataChannel("oai-events");
      c.channel.addEventListener("message",event=>onEvent(c,event.data));
      c.channel.addEventListener("close",()=>{if(connection===c && !c.closing)end("Lidhja u mbyll. / Voice connection closed.");});
      await c.peer.setLocalDescription(await c.peer.createOffer());
      await new Promise((resolve,reject)=>{
        if(c.peer.iceGatheringState==="complete")return resolve();
        const timer=setTimeout(()=>{c.peer.removeEventListener("icegatheringstatechange",changed);reject(new Error("Microphone connection timed out. Try again."));},10000);
        function changed(){if(c.peer.iceGatheringState==="complete"){clearTimeout(timer);c.peer.removeEventListener("icegatheringstatechange",changed);resolve();}}
        c.peer.addEventListener("icegatheringstatechange",changed);
      });
      if(connection!==c || c.closing)return;
      // Do not abort this creation request on End: a late answer must still be hung up.
      const result=await post("session",{sdp:c.peer.localDescription.sdp,context:sheet});
      c.owner={session_id:result.session_id,token:result.token};
      if(connection!==c || c.closing){hangup(c);return;}
      if(JSON.stringify(context())!==JSON.stringify(sheet)){end("Fleta ndryshoi. Niseni bisedën sërish. / The sheet changed. Start voice again.");return;}
      await c.peer.setRemoteDescription({type:"answer",sdp:result.sdp});
      c.timer=setInterval(()=>tick(c),200);
      c.startTimer=setTimeout(()=>{if(!c.started && connection===c)end("Zëri nuk u lidh. Provoni përsëri. / Voice did not connect. Try again.");},15000);
    }catch(error){
      if(connection===c && !c.closing){
        const message=error.name==="NotAllowedError"?"Lejoni mikrofonin në shfletues. / Allow microphone access, then try again.":error.message;
        end(message);
      }
      if(connection!==c || c.closing)c.stream?.getTracks().forEach(t=>t.stop());
    }
  }
  root.ProductionSheetVoice={update,end};
  el("productionSheetVoiceStart").addEventListener("click",()=>void start());
  el("productionSheetVoiceEnd").addEventListener("click",()=>end());
  el("productionSheetVoiceMute").addEventListener("click",()=>{
    const c=connection;if(!c?.started || c.closing)return;c.muted=!c.muted;c.lastActivity=Date.now();
    c.stream.getAudioTracks().forEach(t=>{t.enabled=!c.muted;});
    send(c,c.muted?"session.input_audio.mute":"session.input_audio.unmute");
    el("productionSheetVoiceMute").textContent=c.muted?"Aktivizo mikrofonin / Unmute":"Hesht mikrofonin / Mute";
    el("productionSheetVoiceMute").setAttribute("aria-pressed",String(c.muted));
  });
  el("productionSheetVoicePlay").addEventListener("click",()=>{
    const c=connection;if(!c || c.closing)return;c.audioContext.resume();
    c.audio?.play().then(()=>{el("productionSheetVoicePlay").hidden=true;}).catch(()=>status("Lejoni audion në shfletues. / Allow audio playback."));
  });
  for(const type of ["pointerdown","keydown","input"])dialog.addEventListener(type,()=>{if(connection)connection.lastActivity=Date.now();});
  root.addEventListener("pagehide",()=>end("",true));
  update();
})(window);
