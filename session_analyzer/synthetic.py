"""Explicitly synthetic fixtures from the current schema; no AI or image packages."""
from __future__ import annotations

from contextlib import closing
import html
import json
from pathlib import Path
import sqlite3

from .source import literals

TOPICS = [
    ("멀티모달 강의 맥락", "화면과 음성을 하나의 시간축으로 연결하기"),
    ("화면 변화와 대표 프레임", "P / C / R 비교를 통한 저장 효율화"),
    ("음성 전사와 문장 청크", "발화 경계에서 검색 가능한 문맥 만들기"),
    ("Hybrid Retrieval", "BM25와 의미 검색의 순위를 RRF로 결합"),
    ("질문에 필요한 맥락 선택", "대표 화면 · 질문 스냅샷 · 과거 전사"),
]


def slide(index: int, timestamp: int, snapshot: bool = False) -> str:
    title, subtitle = TOPICS[(index // 36) % len(TOPICS)]
    chart = "".join(f'<rect x="{760+i*65}" y="{470-height}" width="38" height="{height}" rx="5" fill="{color}"/>'
                    for i, (height, color) in enumerate([(110,"#cbd9fa"),(180,"#a7bff3"),(140,"#8da8ed"),(260,"#416fe0"),(220,"#7057c3")]))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<rect width="1280" height="720" fill="#fff"/><rect width="12" height="720" fill="#416fe0"/>
<g font-family="Segoe UI,Malgun Gothic,sans-serif"><text x="70" y="73" font-size="17" font-weight="700" letter-spacing="3" fill="#416fe0">CLUE / LECTURE LAB</text>
<text x="70" y="164" font-size="42" font-weight="700" fill="#20242b">{html.escape(title)}</text>
<text x="70" y="208" font-size="21" fill="#8491a5">{html.escape(subtitle)}</text>
<line x1="70" x2="1210" y1="246" y2="246" stroke="#e4e9f0"/>
<rect x="70" y="291" width="565" height="270" rx="16" fill="#f4f7fe"/>
<text x="102" y="340" font-size="22" fill="#416fe0" font-weight="600">화면 + 음성 + 질문</text>
<text x="102" y="397" font-size="20" fill="#5e6e85">01   공통 세션 시계로 관측 연결</text>
<text x="102" y="447" font-size="20" fill="#5e6e85">02   질문에 필요한 근거 선택</text>
<text x="102" y="497" font-size="20" fill="#5e6e85">03   선택 기록으로 동작 확인</text>
<line x1="730" x2="1120" y1="471" y2="471" stroke="#c8d3e5"/>{chart}
<text x="755" y="521" font-size="16" fill="#8491a5">개념 예시 · 성능 수치 아님</text>
<rect x="70" y="612" width="1140" height="57" rx="9" fill="#fff3dd"/>
<text x="92" y="649" font-size="20" fill="#93631c" font-weight="600">SYNTHETIC · 합성 화면 · 실제 강의/측정 아님</text>
<text x="840" y="649" font-size="16" fill="#93631c">{timestamp/1000:.1f}s / {'질문 스냅샷' if snapshot else '정기 대표'}</text></g></svg>'''


def generate(folder: Path, observations: int = 180) -> Path:
    if observations < 1 or observations > 200000:
        raise ValueError("관측 수는 1~200000 사이로 지정해 주세요.")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "frames").mkdir()
    (folder / "SYNTHETIC_SESSION.json").write_text(json.dumps({"synthetic": True,
        "description": "UI/기능 검증용 생성 데이터. 실제 강의·성능 측정·모델 실행 결과가 아닙니다.",
        "schema_source": "context_qa/store.py SCHEMA", "generator": "scripts.generate_analyzer_demo"}, ensure_ascii=False, indent=2), encoding="utf-8")
    sid = "SYNTHETIC_멘토링_데모"
    with closing(sqlite3.connect(folder / "metadata.db")) as conn, conn:
        schema = literals("context_qa/store.py")
        conn.executescript(schema["SCHEMA"])
        for table, additions in schema.get("ADDED_COLUMNS", {}).items():
            existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            for column, definition in additions:
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

        def insert(table, values):
            keys = ", ".join(values)
            conn.execute(f"INSERT INTO {table} ({keys}) VALUES ({','.join('?' for _ in values)})", list(values.values()))

        def frame(fid, seq, time, source, decision, group, index):
            relative = f"frames/{fid}.svg"
            (folder / relative).write_text(slide(index, time, source != "periodic"), encoding="utf-8")
            insert("frames", dict(session_id=sid, frame_id=fid, image_path=relative,
                frame_type="Unknown" if source != "periodic" else ["Text","Visual","Unknown"][(index//12)%3],
                source=source, decision=decision, group_id=group, first_capture_seq=seq, first_timestamp_ms=time))

        fid, group = None, None
        seq = 0
        periodic_ids = []
        for index in range(observations):
            seq += 1
            time = 1000 + index * 5000
            decision = "new_group" if index % 36 == 0 else "new_representative" if index % 12 == 0 else "reused"
            if decision == "new_group":
                group = f"g{index//36+1:03}"
            if decision != "reused":
                fid = f"f{index:06}"
                periodic_ids.append((time, fid))
                frame(fid, seq, time, "periodic", decision, group, index)
            insert("frame_history", dict(session_id=sid, capture_seq=seq, frame_id=fid,
                timestamp_ms=time, source="periodic", decision=decision, group_id=group))
        duration = 1000 + (observations - 1) * 5000
        segment_ids = []
        for index, time in enumerate(range(1500, duration, 7000)):
            topic = TOPICS[(index//25)%len(TOPICS)][0]
            segment_id = f"s{index:05}"
            insert("transcripts", dict(session_id=sid, segment_id=segment_id, start_ms=time,
                end_ms=min(time+4200,duration), text=f"[합성 전사] {topic}의 핵심은 질문에 필요한 근거를 시간과 함께 연결하는 것입니다. 예시 발화 {index+1}."))
            segment_ids.append((time, segment_id))
        for index in range(0,len(segment_ids),3):
            parts=segment_ids[index:index+3]
            chunk_id=f"c{index//3:04}"
            start=parts[0][0]
            end=min(parts[-1][0]+4200,duration)
            insert("text_chunks",dict(session_id=sid,chunk_id=chunk_id,start_ms=start,end_ms=end,
                text=f"[합성 RAG 청크] {TOPICS[(index//25)%len(TOPICS)][0]}. 세션 공통 시각과 원본 전사 ID를 보존합니다.",segment_ids=json.dumps([p[1] for p in parts])))
            if index%9!=0:
                insert("chunk_embeddings",dict(session_id=sid,chunk_id=chunk_id,
                    embedding_model="SYNTHETIC / MiniLM placeholder · 모델 실행 없음",dim=384,vector=bytes(384*8)))
        questions = [
            ("current_visual", "지금 화면에서 P/C/R은 각각 무엇을 의미하나요?", 1, "completed"),
            ("general", "그렇다면 대표 프레임을 재사용하는 이유는 무엇인가요?", 1, "completed"),
            ("past", "아까 설명한 공통 세션 시계는 무엇인가요?", 2, "completed"),
            ("past_visual", "앞에서 보았던 그래프를 지금 화면과 비교해 주세요.", 2, "completed"),
            ("dispute_answer", "이전 답변에서는 모든 화면을 저장한다고 했잖아요?", 1, "completed"),
            ("dispute_lecture", "교수님 설명과 다른데 음성 중첩은 왜 필요한가요?", None, "failed"),
            ("general", "검색 점수까지 설명해 주세요.", None, "cancelled"),
            ("current_visual", "이 화면을 다시 설명해 주세요.", None, "unanswered"),
        ]
        for index,(kind,text,tag,status) in enumerate(questions):
            submit=max(2,round(duration*[0.15,0.20,0.73,0.78,0.82,0.86,0.90,0.94][index]))
            started=max(0,submit-4500)
            qid=f"q{index+1:03}"
            input_seq=seq+1
            image_ids=[]
            for source,time in (("question_start",started),("question_submit",submit)):
                seq+=1
                snapshot_id=f"{qid}_{source}"
                image_ids.append(snapshot_id)
                frame(snapshot_id,seq,time,source,"question_snapshot",None,round(time/5000))
                insert("frame_history",dict(session_id=sid,capture_seq=seq,frame_id=snapshot_id,timestamp_ms=time,source=source,decision="question_snapshot",group_id=None))
            insert("questions",dict(session_id=sid,question_id=qid,text="[합성 질문] "+text,input_started_ms=started,
                submitted_ms=submit,input_capture_seq=input_seq,submit_capture_seq=seq,question_type=kind))
            if status=="unanswered":
                continue
            eligible=[fid for time,fid in periodic_ids if time<=submit]
            selected=eligible[-2:]+image_ids
            selected_segments=[sid for time,sid in segment_ids if time+4200<=submit][-5:]
            if kind.startswith("past"):
                selected=list(dict.fromkeys(eligible[:1]+selected))
                selected_segments=list(dict.fromkeys([sid for time,sid in segment_ids if time+4200<=submit][:3]+selected_segments))
            complete=status=="completed"
            insert("answers",dict(session_id=sid,question_id=qid,
                answer="[합성 답변] 화면과 음성은 같은 세션 시계에 기록됩니다. 질문에 사용된 이미지와 전사의 ID가 저장되어 근거를 확인할 수 있습니다. 이 문장은 UI 시연용이며 실제 모델 응답이 아닙니다." if complete else "[합성 부분 답변] 기록을 확인하면…" if status=="cancelled" else "",
                context_tag=tag,used_frame_ids=json.dumps(selected if complete else []),used_segment_ids=json.dumps(selected_segments if complete else []),
                first_response_latency_ms=640+index*137 if complete else None,elapsed_ms=2100+index*430,
                status=status,error_code="SYNTHETIC_TIMEOUT" if status=="failed" else None,
                input_tokens=1500+index*390 if complete else None,output_tokens=95+index*23 if complete else None,
                model="SYNTHETIC / demo-llm · 모델 실행 없음",strategy="full",rag_search_ms=18+index*3 if kind in {"past","past_visual","dispute_lecture"} else None,retries=1 if status=="failed" else 0))
    return folder
