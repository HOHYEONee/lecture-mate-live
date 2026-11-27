"use strict";
(() => {
  const D = JSON.parse(document.getElementById("session-data").textContent);
  const S = D.summary;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "미기록").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const n = (x, digits = 0) => x == null ? "—" : Number(x).toLocaleString("ko-KR", {maximumFractionDigits: digits});
  const seconds = (x) => x == null ? "미기록" : `${n(x / 1000, 2)}초`;
  const clock = (x, precise = false) => {
    if (x == null) return "미기록";
    const ms = Math.max(0, Math.round(x));
    const s = Math.floor(ms / 1000);
    const value = `${String(Math.floor(s / 3600)).padStart(2,"0")}:${String(Math.floor(s / 60) % 60).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
    return value + (precise ? `.${String(ms % 1000).padStart(3,"0")}` : "");
  };
  const bytes = (x) => x == null ? "미기록" : x < 1048576 ? `${n(x / 1024, 1)} KB` : `${n(x / 1048576, 1)} MB`;
  const F = new Map(D.frames.map((f) => [f.frame_id, f]));
  const T = new Map(D.transcripts.map((t) => [t.segment_id, t]));
  const Q = new Map(D.questions.map((q) => [q.question_id, q]));
  const H = D.history.filter((h) => h.source === "periodic" && h.timestamp_ms != null);
  const colors = {new_group:"#416fe0",new_representative:"#8b63ce",reused:"#b6c5e1",Text:"#416fe0",Visual:"#8b63ce",Unknown:"#c4cddc"};
  const sourceLabel = {periodic:"정기 대표 프레임",question_start:"질문 시작 스냅샷",question_submit:"질문 제출 스냅샷"};
  const statusLabel = {completed:"정상 완료",cancelled:"취소",failed:"실패",unanswered:"답변 미기록",unknown:"상태 미기록"};
  const selectedTime = D.questions.find((q) => q.submitted_ms != null)?.submitted_ms ?? S.observed_range.start_ms ?? 0;
  const state = {tab:"",done:new Set(),time:selectedTime,gallery:"all",group:D.groups[0]?.group_id,audio:D.transcripts[0]?.segment_id,question:D.questions[0]?.question_id};
  const badge = (text, cls="") => `<span class="badge ${cls}">${esc(text)}</span>`;
  const status = (q) => `<span class="status ${esc(q.status)}"></span>${esc(statusLabel[q.status] || q.status)}`;
  const empty = (text="해당 데이터가 없습니다.") => `<div class="empty">${esc(text)}</div>`;
  const notice = (text, warm=false) => `<div class="notice${warm ? " warm" : ""}">${text}</div>`;
  const metric = (label, value, unit="", detail="", accent=false) => `<div class="metric${accent ? " accent" : ""}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}<span>${esc(unit)}</span></div><div class="detail">${esc(detail)}</div></div>`;
  const heading = (title, description, index) => `<div class="section-heading"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div><span class="section-num">MODULE / ${index}</span></div>`;
  const panel = (title, body, aside="") => `<div class="panel"><div class="panel-head"><h3>${esc(title)}</h3><small>${esc(aside)}</small></div>${body}</div>`;
  const table = (headers, rows) => rows.length ? `<div class="table-scroll"><table><thead><tr>${headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>` : empty();
  const td = (x, cls="") => `<td class="${cls}">${x}</td>`;
  const row = (cells, attrs="") => `<tr ${attrs}>${cells.map((c) => td(c)).join("")}</tr>`;
  function kv(entries) { return `<div class="kv">${entries.map(([k,v]) => `<div>${esc(k)}<strong>${esc(v)}</strong></div>`).join("")}</div>`; }
  function textBlock(text, max=500) {
    if (!text) return `<span class="muted">미기록</span>`;
    if (text.length <= max) return `<div class="body-text">${esc(text)}</div>`;
    return `<div class="body-text">${esc(text.slice(0,max))}…</div><details><summary>전체 내용 펼치기</summary><div class="body-text">${esc(text)}</div></details>`;
  }
  function frameCard(f, label="", extra="") {
    if (!f) return empty(label ? `${label} · 프레임 메타데이터 미기록` : "프레임 메타데이터 미기록");
    return `<article class="frame-card">${f.asset ? `<img src="${esc(f.asset)}" alt="${esc(sourceLabel[f.source] || f.source)} ${esc(clock(f.first_timestamp_ms))}" loading="lazy" decoding="async" data-frame="${esc(f.frame_id)}">` : `<div class="missing">이미지 파일 없음</div>`}<div class="frame-meta">${label ? `<div class="frame-title">${esc(label)}</div>` : ""}${badge(f.frame_type || "미기록")}${badge(sourceLabel[f.source] || "원인 미기록",f.source?.startsWith("question") ? "purple" : "")}<div>${esc(clock(f.first_timestamp_ms, true))} · ${esc(bytes(f.file_bytes))}</div><div>${esc(f.decision)} ${extra}</div><div class="mono">${esc(f.frame_id)} · ${esc(f.group_id || "그룹 없음")}</div></div></article>`;
  }
  // Only the current page is materialized. Full arrays stay in the embedded data/exports.
  function paged(id, items, render, size=24) {
    let page=0;
    const host=$(id);
    function paint() {
      const pages=Math.max(1,Math.ceil(items.length / size));
      page=Math.max(0,Math.min(page,pages-1));
      host.innerHTML=render(items.slice(page*size,(page+1)*size),page*size)+`<div class="pager"><span>${n(items.length)}건 · ${page+1}/${pages}</span><button data-page="prev" ${page===0 ? "disabled" : ""}>이전</button><button data-page="next" ${page===pages-1 ? "disabled" : ""}>다음</button></div>`;
    }
    host.onclick=(event) => { const b=event.target.closest("[data-page]"); if (!b) return; page+=b.dataset.page==="next" ? 1 : -1; paint(); };
    paint();
  }
  function observedFrame(time) {
    let lo=0,hi=H.length;
    while(lo<hi) { const mid=(lo+hi)>>>1; if(H[mid].timestamp_ms<=time)lo=mid+1;else hi=mid; }
    const h=lo ? H[lo-1] : null;
    return {h,f:h ? F.get(h.frame_id) : null};
  }
  function bars(entries, formatter=(v)=>n(v), maximum=null) {
    if(!entries.length)return empty();
    const max=maximum || Math.max(1,...entries.map((x)=>x.value || 0));
    return `<div class="bar-chart">${entries.map((x)=>`<div class="bar-row"><span class="bar-label" title="${esc(x.label)}">${esc(x.label)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,(x.value||0)/max*100)}%;background:${x.color || colors[x.label] || "#416fe0"}"></div></div><span class="numeric">${esc(formatter(x.value))}</span></div>`).join("")}</div>`;
  }
  function histogram(values) {
    values=values.filter((v)=>v!=null && v>=0);
    if(!values.length)return empty("지속 시간 기록이 없습니다.");
    const edges=[1000,3000,5000,10000,20000,Infinity], counts=edges.map(()=>0);
    for(const v of values)counts[edges.findIndex((edge)=>v<edge)]++;
    return bars(counts.map((count,i)=>({label:["1초 미만","1~3초","3~5초","5~10초","10~20초","20초 이상"][i],value:count})),(v)=>`${n(v)}건`);
  }
  function timeline(hostId,onSelect,frameOnly=false) {
    const host=$(hostId), full=S.observed_range;
    let start=full.start_ms??0,end=Math.max(start+1,full.end_ms??1),value=Math.max(start,Math.min(end,state.time));
    host.innerHTML=`<canvas class="timeline" role="img" aria-label="세션 공통 시계 타임라인. 아래 슬라이더와 초 입력으로 시점을 선택할 수 있습니다."></canvas><div class="timeline-controls"><output>${clock(value,true)}</output><input type="range" aria-label="세션 시점" min="${start}" max="${end}" value="${value}" step="1"><label>시점(초) <input type="number" aria-label="선택 시점 초" step="0.001" min="${start/1000}" max="${end/1000}" value="${value/1000}"></label></div><div class="zoom"><span>표시 구간(초)</span><input type="number" aria-label="표시 시작 초" value="${start/1000}" min="0" step="1"><span>~</span><input type="number" aria-label="표시 종료 초" value="${end/1000}" min="0" step="1"><button data-zoom>구간 확대</button><button data-reset>전체 보기</button><span>시간축을 클릭하거나 키보드로 슬라이더를 조절하세요.</span></div>`;
    const canvas=host.querySelector("canvas"),range=host.querySelector("[type=range]"),inputs=host.querySelectorAll("[type=number]"),out=host.querySelector("output");
    let width=900,height=frameOnly?108:190;
    function draw() {
      width=canvas.clientWidth || 900;
      const ratio=window.devicePixelRatio || 1;
      canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);canvas.style.height=`${height}px`;
      const c=canvas.getContext("2d");c.scale(ratio,ratio);c.clearRect(0,0,width,height);
      const left=91,right=width-16,plot=right-left;
      const x=(t)=>left+(t-start)/(end-start)*plot;
      c.font='10px "Segoe UI","Malgun Gothic",sans-serif';
      for(let i=0;i<=5;i++){ const time=start+(end-start)*i/5,px=x(time);c.strokeStyle="#edf1f7";c.beginPath();c.moveTo(px,10);c.lineTo(px,height-30);c.stroke();c.fillStyle="#90a0b4";c.textAlign=i===5?"right":"left";c.fillText(clock(time),px,height-8); }
      const lanes=frameOnly?[{label:"P/C/R 판정",y:32,rows:H,start:"timestamp_ms",color:(r)=>colors[r.decision]||"#c4cddc"}]:[
        {label:"대표 프레임",y:30,rows:D.frames.filter((f)=>f.source==="periodic"),start:"first_timestamp_ms",color:()=>"#416fe0"},
        {label:"음성 전사",y:65,rows:D.transcripts,start:"start_ms",end:"end_ms",color:()=>"#44a38e"},
        {label:"RAG 청크",y:100,rows:D.chunks,start:"start_ms",end:"end_ms",color:()=>"#9676d6"},
        {label:"질문 시작/제출",y:135,rows:D.questions.flatMap((q)=>[{time:q.input_started_ms,submit:false},{time:q.submitted_ms,submit:true}]),start:"time",color:(r)=>r.submit?"#d09a41":"#e5cda2"}
      ];
      for(const lane of lanes){
        c.textAlign="left";c.fillStyle="#7c8aa0";c.fillText(lane.label,0,lane.y+4);c.fillStyle="#f4f6fb";c.fillRect(left,lane.y-7,plot,14);
        // Bound drawing calls by pixels/color. Overlapping marks share one pixel.
        const marks=new Map();
        for(const item of lane.rows){const a=item[lane.start],b=lane.end?item[lane.end]:a;if(a==null||b==null||b<start||a>end)continue;const lo=Math.max(left,Math.floor(x(Math.max(start,a)))),hi=Math.min(right,Math.max(lo+2,Math.ceil(x(Math.min(end,b)))));const color=lane.color(item);for(let px=lo;px<hi;px++)marks.set(`${px}:${color}`,{px,color});}
        for(const {px,color} of marks.values()){c.fillStyle=color;c.fillRect(px,lane.y-5,2,10);}
      }
      if(value>=start&&value<=end){c.strokeStyle="#263f6f";c.setLineDash([3,3]);c.beginPath();c.moveTo(x(value),7);c.lineTo(x(value),height-26);c.stroke();c.setLineDash([]);}
    }
    function select(t){value=Math.max(start,Math.min(end,t));range.value=value;inputs[0].value=(value/1000).toFixed(3);out.textContent=clock(value,true);state.time=value;draw();onSelect(value);}
    canvas.onclick=(e)=>{const rect=canvas.getBoundingClientRect();select(start+Math.max(0,Math.min(1,(e.clientX-rect.left-91)/(width-107)))*(end-start));};
    range.oninput=()=>select(Number(range.value));inputs[0].onchange=()=>select(Number(inputs[0].value)*1000);
    host.querySelector("[data-zoom]").onclick=()=>{const a=Number(inputs[1].value)*1000,b=Number(inputs[2].value)*1000;if(a>=0&&b>a){start=a;end=b;range.min=a;range.max=b;select(value);}};
    host.querySelector("[data-reset]").onclick=()=>{start=full.start_ms??0;end=Math.max(start+1,full.end_ms??1);inputs[1].value=start/1000;inputs[2].value=end/1000;range.min=start;range.max=end;select(value);};
    const observer=new ResizeObserver(()=>{if(canvas.clientWidth)draw();});observer.observe(canvas);draw();
  }
  function overviewDetail(time){
    const {f,h}=observedFrame(time);
    $("overview-frame").innerHTML=frameCard(f,"선택 시점의 대표 화면",h?`· 관측 ${clock(h.timestamp_ms)}`:"");
    const active=D.transcripts.filter((s)=>s.start_ms<=time && s.end_ms>=time);
    const nearby=active.length?active:D.transcripts.filter((s)=>Math.abs(s.start_ms-time)<=15000).slice(0,3);
    const questions=D.questions.filter((q)=>Math.abs((q.submitted_ms??Infinity)-time)<=30000).slice(0,3);
    $("overview-nearby").innerHTML=`<div class="panel-head"><h3>이 시점의 맥락</h3><small>${clock(time,true)}</small></div>${h?`<p class="mini">직전 정기 관측 ${clock(h.timestamp_ms,true)} · ${esc(h.decision)}<br>다음 관측 전까지 화면이 같았다고 확정할 수는 없습니다.</p>`:notice("이 시점 이전 정기 관측이 없습니다.")}
      <div class="divider"></div><h3>강의 전사 ${active.length?"· 해당 구간":"· ±15초 주변"}</h3>${nearby.length?nearby.map((s)=>`<div class="transcript-item"><div class="transcript-time">${clock(s.start_ms)}<br>${clock(s.end_ms)}</div><div class="transcript-body">${esc(s.text)}</div></div>`).join(""):empty("주변 전사 기록이 없습니다.")}
      <div class="divider"></div><h3>주변 질문 <small>±30초</small></h3>${questions.length?questions.map((q)=>`<button class="question-card" data-question="${esc(q.question_id)}"><div class="q-head">${clock(q.submitted_ms)} ${badge(q.question_type||"유형 미기록")}</div><div class="q-text">${esc(q.text)}</div>${status(q)}</button>`).join(""):empty("주변 질문 기록이 없습니다.")}`;
  }
  function renderOverview(){
    $("overview").innerHTML=heading("하나의 시간축, 연결된 강의 맥락","화면 관측부터 질문 응답까지, 세션에 남은 기록을 함께 살펴봅니다.","01")+
      `<div class="metrics">${metric("정기 프레임 관측",n(S.periodic_observations),"회","frame_history · periodic")}${metric("저장된 정기 대표 프레임",n(S.periodic_frames),"장",`질문 스냅샷 ${n((S.snapshot_start??0)+(S.snapshot_submit??0))}장 별도`)}${metric("정기 프레임 저장 감소율",n(S.storage_reduction_pct,1),"%","질문 스냅샷 제외",true)}${metric("전사된 발화",n(S.transcript_count),"개",`RAG 청크 ${n(S.chunk_count)}개`)}${metric("질문 / 정상 완료",`${n(S.question_count)} / ${n(S.statuses.completed??0)}`,"","완료·취소·실패 분리 집계")}${metric("전체 응답 중앙값",n(S.completed_total_ms.median==null?null:S.completed_total_ms.median/1000,2),"초",`첫 텍스트 ${seconds(S.completed_first_ms.median)} · 정상 완료만`)}</div>`+
      panel("통합 세션 타임라인",`<div class="legend"><span>대표 프레임</span><span style="--dot:#44a38e">음성 전사</span><span style="--dot:#9676d6">RAG 청크</span><span style="--dot:#d09a41">질문 시작 / 제출</span></div><div id="overview-timeline"></div>`,"공통 세션 시계 · 일시정지 제외")+
      `<div class="grid2"><div class="panel"><div class="panel-head"><h3>화면과 설명의 연결</h3><small>frame_history 기준</small></div><div id="overview-frame"></div></div><div class="panel" id="overview-nearby"></div></div>`+
      panel("기록의 범위",kv([["관측 데이터 범위",`${clock(S.observed_range.start_ms)} ~ ${clock(S.observed_range.end_ms)}`],["정확한 실행 시작 / 종료","미기록"],["기록된 LLM 모델",S.models.join(", ")||"미기록"],["데이터 진단",`${D.warnings.length}건`]])+notice("관측 범위는 전체 실행 시간이 아닙니다. 분류 분포는 정확도가 아니며, SSIM 점수·전사 지연·답변 정답률·API 비용은 추가 근거 없이 산출하지 않습니다."));
    timeline("overview-timeline",overviewDetail);overviewDetail(state.time);
  }
  function renderGallery(){
    const filters={all:()=>true,Text:(f)=>f.frame_type==="Text",Visual:(f)=>f.frame_type==="Visual",Unknown:(f)=>f.frame_type==="Unknown",new_group:(f)=>f.decision==="new_group",new_representative:(f)=>f.decision==="new_representative",question_start:(f)=>f.source==="question_start",question_submit:(f)=>f.source==="question_submit"};
    const items=D.frames.filter(filters[state.gallery]||filters.all);
    paged("frame-gallery",items,(rows)=>rows.length?`<div class="gallery">${rows.map((f)=>frameCard(f)).join("")}</div>`:empty(),20);
    $("gallery-filters").querySelectorAll("button").forEach((b)=>b.classList.toggle("active",b.dataset.filter===state.gallery));
  }
  function groupDetail(){
    const g=D.groups.find((g)=>g.group_id===state.group);
    if(!g){$("group-detail").innerHTML=empty();return;}
    $("group-detail").innerHTML=`<div class="panel-head"><h3>선택 그룹의 대표 변경과 재사용</h3><span class="mono">${esc(g.group_id)}</span></div>${kv([["그룹 생성 관측",clock(g.created_ms,true)],["관측 수",`${n(g.observations)}회`],["대표 이미지",`${n(g.frame_ids.length)}장`],["대표 변경",`${n(g.changes.filter((x)=>x.decision==="new_representative").length)}회`]])}<div id="group-images"></div><details><summary>대표 변경 시점 / 재사용 구간 펼치기</summary><div id="group-events"></div></details><p class="content-note">재사용 구간의 끝은 다음 정기 관측 시점입니다. 마지막 관측 이후 구간은 미기록입니다.</p>`;
    paged("group-images",g.frame_ids,(ids)=>`<div class="gallery compact">${ids.map((id)=>frameCard(F.get(id))).join("")}</div>`,6);
    const events=[...g.changes.map((x)=>({...x,start_ms:x.timestamp_ms,end_ms:null})),...g.reuse_intervals.map((x)=>({...x,decision:"reused"}))].sort((a,b)=>(a.start_ms??0)-(b.start_ms??0));
    paged("group-events",events,(rows)=>table(["관측 / 시작","다음 관측","판정","대표 프레임"],rows.map((r)=>row([clock(r.start_ms,true),r.decision==="reused"?clock(r.end_ms,true):"—",esc(r.decision),`<span class="mono">${esc(r.frame_id)}</span>`]))),25);
  }
  function renderFrames(){
    $("frames").innerHTML=heading("반복되는 화면은 재사용하고, 변화는 보존합니다","P: 직전 관측 · C: 현재 화면 · R: 현재 그룹의 대표 화면", "02")+
      `<div class="metrics">${metric("정기 관측",n(S.periodic_observations),"회")}${metric("대표 저장",n(S.periodic_frames),"장")}${metric("재사용",n(S.reused_observations),"회")}${metric("새 그룹 / 그룹 내 새 대표",`${n(S.new_groups)} / ${n(S.new_representatives)}`,"")}${metric("질문 시작 / 제출 스냅샷",`${n(S.snapshot_start)} / ${n(S.snapshot_submit)}`,"장")}${metric("JPEG 저장 용량",bytes(S.jpeg_bytes),"",`${n(S.jpeg_count)}파일 · 누락 ${n(S.missing_images)}개`)}</div>`+
      `<div class="grid2 grid-equal">${panel("P/C/R 판정",`<div class="step-flow"><div class="step"><strong>P ↔ C</strong>새 그룹 여부<br>기본 임계값 ${esc(D.defaults.pc_threshold)}</div><div class="arrow">→</div><div class="step"><strong>R ↔ C</strong>대표 변경 여부<br>기본 임계값 ${esc(D.defaults.rc_threshold)}</div><div class="arrow">→</div><div class="step"><strong>기존 대표</strong>유사 화면 재사용<br>reused</div></div>`+notice("현재 코드 기본값입니다. 세션 실행 임계값과 개별 P/C·R/C SSIM 점수는 DB에 미기록입니다."))}${panel("정기 대표 프레임 분류 분포",bars(Object.entries(S.classification).map(([label,value])=>({label,value})),(v)=>`${n(v)}장`)+notice("질문 스냅샷은 분류를 거치지 않는 Unknown이므로 이 분포에서 제외합니다. 정답 레이블이 없어 분류 정확도는 산출할 수 없습니다."))}</div>`+
      panel("관측 판정 타임라인",`<div class="legend"><span>new_group</span><span style="--dot:#8b63ce">new_representative</span><span style="--dot:#b6c5e1">reused</span></div><div id="frame-timeline"></div><div id="frame-observations"></div>`, `캡처 간격 평균 ${seconds(S.capture_interval_ms.mean)} · 중앙 ${seconds(S.capture_interval_ms.median)} · 최대 ${seconds(S.capture_interval_ms.max)}`)+
      panel("화면 그룹",`<div id="group-list"></div><div class="divider"></div><div id="group-detail"></div>`)+
      panel("저장 이미지 갤러리",`<div class="filters" id="gallery-filters">${[["all","전체"],["Text","Text"],["Visual","Visual"],["Unknown","Unknown"],["new_group","New Group"],["new_representative","New Representative"],["question_start","Question Start"],["question_submit","Question Submit"]].map(([key,label])=>`<button data-filter="${key}">${label}</button>`).join("")}</div><div id="frame-gallery"></div>`, `전체 이미지 파일 ${bytes(S.image_bytes)} · 클릭하여 확대`);
    timeline("frame-timeline",(time)=>{const {h}=observedFrame(time);if(h?.group_id){state.group=h.group_id;groupDetail();}},true);
    paged("frame-observations",D.history,(rows)=>table(["관측 시각","캡처 번호","프레임 ID","그룹 ID","원인","판정"],rows.map((r)=>row([clock(r.timestamp_ms,true),n(r.capture_seq),`<span class="mono">${esc(r.frame_id)}</span>`,`<span class="mono">${esc(r.group_id)}</span>`,esc(r.source),badge(r.decision,r.decision==="new_representative"?"purple":"")]))),30);
    paged("group-list",D.groups,(rows)=>table(["그룹","생성 시점","관측 수","대표 수","상세"],rows.map((g)=>row([`<span class="mono">${esc(g.group_id)}</span>`,clock(g.created_ms),n(g.observations),n(g.frame_ids.length),`<button data-group="${esc(g.group_id)}">그룹 보기</button>`]))),8);
    $("gallery-filters").onclick=(e)=>{const b=e.target.closest("[data-filter]");if(b){state.gallery=b.dataset.filter;renderGallery();}};
    groupDetail();renderGallery();
  }
  function audioDetail(){
    const s=T.get(state.audio);
    if(!s){$("audio-detail").innerHTML=empty("선택할 전사 데이터가 없습니다.");return;}
    $("audio-detail").innerHTML=`<div class="panel-head"><h3>발화와 연결된 화면</h3><span class="mono">${esc(s.segment_id)}</span></div>${kv([["발화 시작",clock(s.start_ms,true)],["발화 종료",clock(s.end_ms,true)],["발화 지속",seconds(s.duration_ms)],["직전 정기 관측",clock(s.observation_ms,true)]])}${textBlock(s.text)}<div class="divider"></div>${frameCard(F.get(s.observed_frame_id),s.observation_decision?`관측 판정: ${s.observation_decision}`:"이전 정기 관측 없음")}${notice("프레임 최초 저장 시각이 아닌, 발화 시작 이전의 마지막 정기 frame_history 관측을 연결합니다. 관측 이후 실제 화면 변화 여부는 미기록입니다.")}`;
  }
  function filterAudio(){
    const term=$("audio-search").value.toLocaleLowerCase(),start=Number($("audio-start").value||0)*1000,end=$("audio-end").value?Number($("audio-end").value)*1000:Infinity;
    const rows=D.transcripts.filter((s)=>(s.text||"").toLocaleLowerCase().includes(term)&&(s.end_ms??-1)>=start&&(s.start_ms??Infinity)<=end);
    paged("audio-list",rows,(items)=>items.length?items.map((s)=>`<div class="transcript-item${s.segment_id===state.audio?" selected":""}" role="button" tabindex="0" data-segment="${esc(s.segment_id)}"><div class="transcript-time">${clock(s.start_ms)}<br>${clock(s.end_ms)}<br><small>${seconds(s.duration_ms)}</small></div><div><div class="transcript-body">${esc(s.text?.slice(0,700))}${s.text?.length>700?"…":""}</div><span class="mono">${esc(s.segment_id)}</span></div></div>`).join(""):empty("검색 조건에 맞는 발화가 없습니다."),18);
  }
  function renderAudio(){
    $("audio").innerHTML=heading("강의의 설명을 시간과 함께 기록합니다","발화 시각과 화면 관측 기록을 연결하고, 전사 결과를 탐색합니다.","03")+
      `<div class="metrics">${metric("전사된 발화",n(S.transcript_count),"개")}${metric("발화 지속 중앙값",n(S.utterance_duration_ms.median==null?null:S.utterance_duration_ms.median/1000,2),"초")}${metric("전사 시작",clock(S.transcript_range.start_ms),"")}${metric("전사 끝",clock(S.transcript_range.end_ms),"")}${metric("전사 사이 공백",n(S.transcript_gaps),"구간","침묵인지 실패인지 판단 불가")}${metric("전사 완료 지연","미기록","","start_ms/end_ms는 발화 시각")}</div>`+
      panel("음성 처리 흐름 · 코드 기본값",`<div class="step-flow"><div class="step"><strong>01 / CAPTURE</strong>WASAPI Loopback<br>시스템 출력 오디오</div><div class="arrow">→</div><div class="step"><strong>02 / CHUNK</strong>신규 ${esc(D.defaults.audio_period_seconds)}초 + 직전 ${esc(D.defaults.audio_overlap_seconds)}초 중첩</div><div class="arrow">→</div><div class="step"><strong>03 / ASR</strong>${esc(D.defaults.asr_model)}<br>GPU float16 / CPU int8</div><div class="arrow">→</div><div class="step"><strong>04 / ASSEMBLE</strong>경계 ${esc(D.defaults.boundary_holdback_ms)}ms 보류<br>중복 단어 조정</div></div>`+notice("현재 코드의 auto 모드는 GPU 우선, 실패하면 같은 모델을 CPU로 실행합니다. 실제 세션의 ASR 모델·장치·정밀도는 metadata.db에 미기록입니다."))+
      `<div class="grid2"><div class="panel"><div class="panel-head"><h3>전사 탐색</h3><a href="transcripts.txt" download>전체 TXT ↓</a></div><input id="audio-search" class="search" placeholder="전사 내용 검색" aria-label="전사 내용 검색"><div class="zoom"><label>시작(초) <input id="audio-start" type="number" min="0" placeholder="0"></label><label>종료(초) <input id="audio-end" type="number" min="0" placeholder="전체"></label><button id="audio-filter">구간 적용</button></div><div id="audio-timeline"></div><div id="audio-list"></div></div><div class="panel" id="audio-detail"></div></div>`+
      `<div class="grid2 grid-equal">${panel("발화 지속 시간 분포",histogram(D.transcripts.map((s)=>s.duration_ms)))}${panel("전사 사이의 공백",`<div id="audio-gaps"></div>`+notice("겹치는 발화를 합친 범위 사이의 공백입니다. 침묵·수집 중단·전사 실패를 구별할 추가 기록은 없습니다."))}</div>`;
    timeline("audio-timeline",(time)=>{const {segment_id}=D.transcripts.find((s)=>s.start_ms<=time&&s.end_ms>=time)||{};if(segment_id){state.audio=segment_id;audioDetail();}});
    $("audio-search").oninput=filterAudio;$("audio-filter").onclick=filterAudio;
    paged("audio-gaps",D.gaps,(rows)=>table(["공백 시작","공백 끝","길이"],rows.map((r)=>row([clock(r.start_ms),clock(r.end_ms),seconds(r.duration_ms)]))),8);
    filterAudio();audioDetail();
  }
  function contextDetail(){
    const q=Q.get(state.question),host=$("question-detail");
    if(!q){host.innerHTML=empty("질문 기록이 없습니다. RAG 청크와 다른 모듈은 계속 탐색할 수 있습니다.");return;}
    const frameIds=q.used_frame_ids,segments=q.used_segment_ids;
    host.innerHTML=panel(`질문 ${String(q.order).padStart(2,"0")} · ${q.question_type||"유형 미기록"}`,`<div class="context-title"><div class="lead">${esc(q.text)}</div>${badge(statusLabel[q.status]||q.status,q.status==="completed"?"green":"")}</div><span class="mono">${esc(q.question_id)}</span>${kv([["입력 시작",clock(q.input_started_ms,true)],["질문 제출",clock(q.submitted_ms,true)],["대화 주제",q.context_tag==null?"미기록":`#${q.context_tag}`],["RAG 검색",q.rag_performed?`${n(q.rag_search_ms)}ms · 수행 시간 기록`:"미기록 / 미수행 여부 확인 불가"]])}${["failed","cancelled"].includes(q.status)?notice("현재 저장 코드는 실패·취소 시 사용 맥락 ID를 빈 배열로 기록합니다. 빈 배열만으로 실제 요청에 맥락이 없었다고 판단할 수 없습니다.",true):""}`)+
      panel("실제 사용된 이미지 맥락",`<div class="content-note">answers.used_frame_ids 순서를 유지합니다. 질문 시작·제출 스냅샷을 정기 대표 화면과 구분합니다.</div><div id="used-images"></div><details><summary>같은 시점 이전의 선택되지 않은 이미지 비교</summary><div id="unused-images"></div></details>`)+
      panel("실제 사용된 음성 맥락",`<div class="content-note">answers.used_segment_ids의 저장 순서입니다. 최근/과거 구분은 코드 기본 최근 범위(${seconds(D.defaults.recent_window_ms)})에 따른 추정이며 실제 선택 역할은 미기록입니다.</div><div id="used-segments"></div>`)+
      panel("같은 대화 주제의 이전 질의응답",notice("context_tag가 같은 정상 완료 이전 질문을 연결합니다. 실제 LLM에 전달된 이전 질의응답 ID·순서는 DB에 미기록입니다.")+`<div id="related-questions"></div>`)+
      panel("최종 답변과 응답 성능",`<div class="answer">${textBlock(q.answer,800)}</div>${kv([["첫 텍스트 응답",seconds(q.first_response_latency_ms)],["전체 처리",seconds(q.elapsed_ms)],["입력 / 출력 토큰",`${n(q.input_tokens)} / ${n(q.output_tokens)}`],["사용 모델",q.model||"미기록"],["맥락 선택 전략",q.strategy||"미기록"],["재시도",n(q.retries)],["오류 코드",q.error_code||"미기록"]])}`);
    if(frameIds===null)$("used-images").innerHTML=empty("사용 이미지 ID 기록이 없거나 손상되었습니다.");
    else paged("used-images",frameIds,(ids,offset)=>ids.length?`<div class="gallery compact">${ids.map((id,i)=>{const f=F.get(id),delta=f?.first_timestamp_ms!=null&&q.submitted_ms!=null?q.submitted_ms-f.first_timestamp_ms:null;return frameCard(f,`전달 순서 ${offset+i+1} · ${sourceLabel[f?.source]||id}`,`· 제출까지 ${seconds(delta)}`);}).join("")}</div>`:empty("사용 이미지 목록이 비어 있습니다."),6);
    const selected=new Set(frameIds||[]),unused=D.frames.filter((f)=>!selected.has(f.frame_id)&&q.submitted_ms!=null&&f.first_timestamp_ms!=null&&f.first_timestamp_ms<=q.submitted_ms);
    paged("unused-images",unused,(rows)=>rows.length?`<div class="gallery compact">${rows.map((f)=>frameCard(f,"선택되지 않은 저장 이미지")).join("")}</div>`:empty(),6);
    if(segments===null)$("used-segments").innerHTML=empty("사용 전사 ID 기록이 없거나 손상되었습니다.");
    else paged("used-segments",segments,(ids,offset)=>ids.length?table(["전달 순서 / ID","발화 범위","제출까지 시간 차","범위 추정","전사"],ids.map((id,i)=>{const s=T.get(id),delta=s?.end_ms!=null&&q.submitted_ms!=null?q.submitted_ms-s.end_ms:null,recent=delta!=null&&D.defaults.recent_window_ms!=null?delta>=0&&delta<D.defaults.recent_window_ms:null;return `<tr>${td(`${offset+i+1}<br><span class="mono">${esc(id)}</span>`)}${td(s?`${clock(s.start_ms)}<br>${clock(s.end_ms)}`:"참조 누락")}${td(seconds(delta))}${td(recent===null?"미기록":recent?"최근 범위 · 추정":"최근 범위 밖 · 추정")}${td(s?textBlock(s.text,300):"전사 메타데이터 없음","wrap")}</tr>`;})):empty("사용 전사 목록이 비어 있습니다."),20);
    paged("related-questions",q.related_previous_ids,(ids)=>ids.length?ids.map((id)=>{const p=Q.get(id);return `<details class="related"><summary>${clock(p.submitted_ms)} · ${esc(p.text)}</summary><p>${esc(p.answer)}</p><span class="mono">${esc(id)}</span><button data-question="${esc(id)}">이 질문 상세</button></details>`;}).join(""):empty("같은 주제의 정상 완료 이전 질문이 없습니다."),6);
  }
  function filterQuestions(){
    const type=$("question-type").value,term=$("question-search").value.toLocaleLowerCase();
    const rows=D.questions.filter((q)=>(!type||q.question_type===type)&&(q.text||"").toLocaleLowerCase().includes(term));
    paged("question-list",rows,(rows)=>rows.length?rows.map((q)=>`<button class="question-card${q.question_id===state.question?" active":""}" data-question="${esc(q.question_id)}"><div class="q-head"><span>Q${String(q.order).padStart(2,"0")} · ${clock(q.submitted_ms)}</span><span>${q.context_tag==null?"—":`주제 #${q.context_tag}`}</span></div><div class="q-text">${esc(q.text)}</div><div class="q-head">${badge(q.question_type||"유형 미기록")}${status(q)}</div></button>`).join(""):empty(),12);
  }
  function filterChunks(){
    const term=$("chunk-search").value.toLocaleLowerCase(),rows=D.chunks.filter((c)=>(c.text||"").toLocaleLowerCase().includes(term));
    paged("chunk-list",rows,(rows)=>rows.length?rows.map((c)=>`<details class="related"><summary><strong>${clock(c.start_ms)} ~ ${clock(c.end_ms)}</strong> · ${seconds(c.duration_ms)} · ${c.embedding_exists===true?"임베딩 저장됨":c.embedding_exists===false?"임베딩 없음":"임베딩 미기록"}<span class="mono"> · ${esc(c.chunk_id)}</span></summary>${textBlock(c.text,1000)}${kv([["임베딩 모델",c.embedding_model||"미기록"],["차원",n(c.dim)]])}<div>원본 전사 ${c.segment_ids===null?"미기록":c.segment_ids.map((id)=>`<button class="chip" data-segment="${esc(id)}">${esc(id)}</button>`).join("")}</div></details>`).join(""):empty("RAG 청크 기록이 없습니다."),15);
  }
  function renderContext(){
    const types=["current_visual","general","past","past_visual","dispute_answer","dispute_lecture"];
    $("context").innerHTML=heading("질문에 전달된 근거를 확인합니다","이미지와 전사 ID는 실제 전달 기록입니다. 검색 순위와 대화 이력 전달 여부는 구분해 해석합니다.","04")+
      `<div class="context-layout"><div><div class="toolbar"><select id="question-type" aria-label="질문 유형"><option value="">모든 질문 유형</option>${types.map((t)=>`<option value="${t}">${t}</option>`).join("")}</select></div><input id="question-search" class="search" placeholder="질문 검색" aria-label="질문 검색"><div class="content-note">${n(S.question_count)}개 질문 · ${n(S.orphan_answer_count)}개 질문 누락 답변</div><div id="question-list" class="question-list"></div></div><div id="question-detail" class="question-detail"></div></div>`+
      panel("전사에서 검색 청크로",`<div class="step-flow"><div class="step"><strong>강의 전사</strong>발화 시각 + Segment ID</div><div class="arrow">→</div><div class="step"><strong>문장 기반 청크</strong>원본 Segment ID 연결</div><div class="arrow">→</div><div class="step"><strong>BM25 + MiniLM</strong>어휘 검색 + 의미 검색</div><div class="arrow">→</div><div class="step"><strong>RRF</strong>두 검색 순위의 융합</div><div class="arrow">→</div><div class="step"><strong>질문 맥락</strong>관련 과거 화면·전사 선택</div></div>`+notice("구조 설명은 현재 코드를 기준으로 합니다. 검색 후보 순위·BM25/의미 유사도/RRF 점수·최종 검색 청크·t*는 metadata.db에 미기록입니다. 임베딩 행의 존재는 저장 기록이며, 분석 중 모델 로드나 검색 재실행은 하지 않습니다.")+`<input id="chunk-search" class="search" placeholder="RAG 청크 검색" aria-label="RAG 청크 검색"><div id="chunk-list"></div>`);
    $("question-type").onchange=filterQuestions;$("question-search").oninput=filterQuestions;$("chunk-search").oninput=filterChunks;
    filterQuestions();contextDetail();filterChunks();
  }
  function pairedBars(rows,keys,labels,formatter){
    if(!rows.length)return empty();
    const max=Math.max(1,...rows.flatMap((r)=>keys.map((key)=>r[key]||0)));
    return `<div class="legend"><span>${esc(labels[0])}</span><span style="--dot:#adbff0">${esc(labels[1])}</span></div><div class="bar-chart" style="margin-top:17px">${rows.map((q)=>`<div class="bar-row"><button class="bar-label" data-question="${esc(q.question_id)}">Q${String(q.order).padStart(2,"0")} · ${esc(statusLabel[q.status]||q.status)}</button><div class="dual-bar">${keys.map((key,i)=>`<div class="bar-track" title="${esc(labels[i])}: ${esc(formatter(q[key]))}"><div class="bar-fill" style="width:${(q[key]||0)/max*100}%;background:${i?"#adbff0":"#416fe0"}"></div></div>`).join("")}</div><span class="numeric">${esc(formatter(q[keys[0]]))}<br>${esc(formatter(q[keys[1]]))}</span></div>`).join("")}</div>`;
  }
  function renderResponse(){
    $("response").innerHTML=heading("질문이 답변이 되기까지","첫 응답은 첫 비어 있지 않은 답변 텍스트 조각을 전달할 때의 시간입니다.","05")+
      `<div class="metrics">${metric("전체 질문",n(S.question_count),"개")}${metric("정상 완료",n(S.statuses.completed??0),"개",`취소 ${n(S.statuses.cancelled??0)} · 실패 ${n(S.statuses.failed??0)}`)}${metric("첫 텍스트 중앙값",n(S.completed_first_ms.median==null?null:S.completed_first_ms.median/1000,2),"초",`유효 표본 ${n(S.completed_first_ms.count)}건`)}${metric("전체 응답 중앙값",n(S.completed_total_ms.median==null?null:S.completed_total_ms.median/1000,2),"초","정상 완료만",true)}${metric("전체 응답 평균",n(S.completed_total_ms.mean==null?null:S.completed_total_ms.mean/1000,2),"초","정상 완료만")}${metric("전체 응답 최대",n(S.completed_total_ms.max==null?null:S.completed_total_ms.max/1000,2),"초","정상 완료만")}</div>`+
      `<div class="grid2 grid-equal">${panel("질문별 응답 시간",`<div id="response-latency"></div>`,"완료·취소·실패 모두 표시 · 단위 초")}${panel("질문별 토큰 사용량",`<div id="response-tokens"></div>`,"기록된 토큰 수만 표시")}</div>`+
      `<div class="grid2 grid-equal">${panel("질문 유형별 전체 응답 중앙값",bars(Object.entries(S.response_by_type).map(([label,s])=>({label,value:s.total_ms.median})),seconds),"정상 완료만 · 유효 표본 기준")}${panel("상태별 질문 수",bars(Object.entries(S.statuses).map(([key,value])=>({label:statusLabel[key]||key,value,color:key==="completed"?"#21876c":key==="failed"?"#d66b58":"#c3a26a"})),(v)=>`${n(v)}건`)+notice("취소·실패·상태 미기록·답변 미기록은 정상 완료 성능 통계에서 제외합니다. 비용과 정답률은 계산하지 않습니다."))}</div>`+
      panel("응답 상세 기록",`<div id="response-table"></div>`+notice("측정 시작은 ContextQAService의 질문 작업 시작입니다. first_response_latency_ms는 태그 필터를 지난 첫 답변 텍스트를 AnswerDelta로 전달하기 직전까지이며, 단순 API 네트워크 지연과 다릅니다. rag_search_ms가 NULL이면 미기록 또는 검색 미수행으로, 재시도·오류·모델 정보도 저장된 값만 표시합니다."));
    paged("response-latency",D.questions,(rows)=>pairedBars(rows,["first_response_latency_ms","elapsed_ms"],["첫 텍스트","전체 처리"],seconds),12);
    paged("response-tokens",D.questions,(rows)=>pairedBars(rows,["input_tokens","output_tokens"],["입력 토큰","출력 토큰"],(v)=>n(v)),12);
    paged("response-table",D.questions,(rows)=>table(["질문","유형","상태","첫 텍스트","전체","RAG(ms)","입력","출력","모델","전략","재시도","오류"],rows.map((q)=>row([`<button data-question="${esc(q.question_id)}">Q${String(q.order).padStart(2,"0")}</button>`,esc(q.question_type),status(q),seconds(q.first_response_latency_ms),seconds(q.elapsed_ms),n(q.rag_search_ms),n(q.input_tokens),n(q.output_tokens),esc(q.model),esc(q.strategy),n(q.retries),esc(q.error_code)]))),30);
  }
  const renderers={overview:renderOverview,frames:renderFrames,audio:renderAudio,context:renderContext,response:renderResponse};
  function switchTab(tab){
    if(!renderers[tab])tab="overview";
    state.tab=tab;
    document.querySelectorAll("[role=tabpanel]").forEach((p)=>p.hidden=p.id!==tab);
    document.querySelectorAll("[data-tab]").forEach((b)=>{b.setAttribute("aria-selected",String(b.dataset.tab===tab));b.tabIndex=b.dataset.tab===tab?0:-1;});
    if(!state.done.has(tab)){renderers[tab]();state.done.add(tab);}
    if(location.hash!==`#${tab}`)history.replaceState(null,"",`#${tab}`);
  }
  document.addEventListener("click",(event)=>{
    const tab=event.target.closest("[data-tab]");if(tab){switchTab(tab.dataset.tab);return;}
    const image=event.target.closest("[data-frame]");if(image){const f=F.get(image.dataset.frame);$("large-image").src=f.asset;$("large-caption").textContent=`${clock(f.first_timestamp_ms,true)} · ${sourceLabel[f.source]} · ${f.frame_id}`;$("image-dialog").showModal();return;}
    const question=event.target.closest("[data-question]");if(question){state.question=question.dataset.question;switchTab("context");contextDetail();filterQuestions();return;}
    const segment=event.target.closest("[data-segment]");if(segment){state.audio=segment.dataset.segment;switchTab("audio");audioDetail();return;}
    const group=event.target.closest("[data-group]");if(group){state.group=group.dataset.group;groupDetail();}
  });
  document.addEventListener("keydown",(event)=>{const item=event.target.closest("[data-segment][role=button]");if(item&&(event.key==="Enter"||event.key===" ")){event.preventDefault();item.click();}const tab=event.target.closest("[data-tab]");if(tab&&["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();const list=[...document.querySelectorAll("[data-tab]")],index=list.indexOf(tab),next=event.key==="Home"?0:event.key==="End"?list.length-1:(index+(event.key==="ArrowRight"?1:-1)+list.length)%list.length;list[next].click();list[next].focus();}});
  $("close-image").onclick=()=>$("image-dialog").close();
  $("export-toggle").onclick=()=>{$("export-panel").hidden=!$("export-panel").hidden;};
  $("diagnostic-toggle").onclick=()=>{$("diagnostics").hidden=!$("diagnostics").hidden;};
  $("session-id").textContent=S.session_id;$("origin").textContent=S.synthetic?"합성 데이터 · 실제 측정 아님":"세션 기록";$("origin").classList.toggle("synthetic",S.synthetic);
  $("range-caption").textContent=`관측 ${clock(S.observed_range.start_ms)} — ${clock(S.observed_range.end_ms)} · 실행 시간과 구분`;
  $("export-panel").innerHTML=[["summary.json","세션 요약 JSON"],["session_analysis.json","전체 분석 JSON"],["frame_summary.csv","프레임 요약 CSV"],["frame_analysis.csv","프레임 관측 CSV"],["frame_metadata.csv","이미지 메타데이터 CSV"],["transcript_analysis.csv","전사 CSV"],["transcripts.txt","전사 TXT"],["transcript_gaps.csv","전사 공백 CSV"],["rag_chunks.csv","RAG 청크 CSV"],["question_context.csv","질문 맥락 CSV"],["response_analysis.csv","응답 성능 CSV"]].map(([file,label])=>`<a download href="${file}">${label} ↓</a>`).join("");
  $("diagnostics").innerHTML=`<h3>데이터 진단 · ${D.warnings.length}건</h3>${D.warnings.length?`<ul class="diagnostic-list">${D.warnings.map((w)=>`<li>${esc(w)}</li>`).join("")}</ul>`:notice("누락된 테이블·열·이미지·ID 참조가 발견되지 않았습니다.")}<details><summary>확인한 스키마</summary>${Object.entries(D.schema).map(([table,cols])=>`<p class="mono">${esc(table)}: ${esc(cols.join(", ")||"테이블 없음")}</p>`).join("")}</details>`;
  window.addEventListener("hashchange",()=>switchTab(location.hash.slice(1)));
  switchTab(location.hash.slice(1)||"overview");
})();
