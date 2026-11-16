# Code Day 7 — Agentic RAG & Retrieval Evaluation

Code chính đặt trong thư mục `src/`, kết nối trực tiếp PostgreSQL + `pgvector` của Day 6 và model Gemini (`gemini-3.5-flash-lite` cho agent, `gemini-embedding-001` cho query embedding).

## Các module

- `export_day6.py`: Export các chunk chiến lược `C` (`307` chunks) từ bảng `rag_chunks` của PostgreSQL Day 6 (`day6_lab`) và `Day06/benchmark/questions.json` sang định dạng Day 7 (`starter/data/converted/` và `data/chunks_day6_c.json`).
- `build_benchmark.py`: Tạo bộ benchmark 30 câu (`data/benchmark_questions.json` và `data/agent_cases.json`), gồm 24 câu có đáp án từ Day 6 và 6 câu thiếu bằng chứng đã kiểm chứng trên toàn bộ 159 trang PDF.
- `retrievers.py`: Cài đặt `PgVectorDenseRetriever` (Day 6), `BM25Retriever`, `TfidfRetriever`, `HybridRRFRetriever`, `CrossFeatureReranker`, `GeminiReranker`, `CorpusNeighborStore` và `pack_context_budget`.
- `retrieval_eval.py`: Chạy benchmark retrieval trên `dev` hoặc `test` và lưu trace vào `results/retrieval/`.
- `agent_pipeline.py`: Chạy `baseline` (qua `starter/agentic.py:run_agent`, 1 tool Dense, `budget=800`) và `selected` (`MultiToolAgenticRAG` dùng native Gemini function calling với `search_hybrid`, `search_lexical`, `get_adjacent_chunks`, `budget=6000`).

## Lệnh chạy tái lập

```powershell
# 1. Kích hoạt môi trường ảo và bật Postgres + pgvector Day 6
.\.venv\Scripts\Activate.ps1
cd ..\Day06\starter; docker compose -p day6_student_starter up -d --wait
cd ..\..\Day07

# 2. Chạy unit tests (offline mocks/fixtures)
python -m unittest discover -s tests -v

# 3. Export dữ liệu Day 6 & dựng benchmark 30 câu
python src/export_day6.py
python src/build_benchmark.py

# 4. Đánh giá các phương pháp retrieval trên dev và chạy baseline vs selected trên test
python src/retrieval_eval.py --split dev --configs tfidf_b800 dense_b800 bm25_b800 hybrid_rrf_b800 hybrid_rrf_b2400 hybrid_rrf_b3600 hybrid_cross_rerank_b3600 hybrid_cross_rerank_neighbor_b4200 hybrid_rrf_neighbor_b4800
python src/retrieval_eval.py --split test --configs dense_b800 hybrid_rrf_neighbor_b4800

# 5. Chạy Agent (baseline vs selected) trên dev và test, sau đó tổng hợp bằng starter/agent_metrics.py
python src/agent_pipeline.py --mode baseline --split dev --model gemini-3.5-flash-lite --output results/agent/baseline_dev_traces.json
python src/agent_pipeline.py --mode selected --split dev --model gemini-3.5-flash-lite --output results/agent/selected_dev_traces.json
python src/agent_pipeline.py --mode baseline --split test --model gemini-3.5-flash-lite --output results/agent/baseline_test_traces.json
python src/agent_pipeline.py --mode selected --split test --model gemini-3.5-flash-lite --output results/agent/selected_test_traces.json

python starter/agent_metrics.py --cases data/agent_cases.json --traces results/agent/baseline_test_traces.json --reviews results/agent/reviews_baseline_test.json --split test --output results/agent/summary_baseline_test.json
python starter/agent_metrics.py --cases data/agent_cases.json --traces results/agent/selected_test_traces.json --reviews results/agent/reviews_selected_test.json --split test --output results/agent/summary_selected_test.json
```

## Thử thách nâng cao (+15 điểm bonus — Multimodal RAG 150 trang)

- `adv_index.py`: Tải (kèm kiểm tra SHA-256) và index toàn bộ 150 trang PDF **NOAA Arctic Report Card 2025** (`Public Domain`, DOI `10.25923/nrzf-j897`) thành `459` chunks (`383` text, `11` table, `65` visual regions + `77` ảnh PNG crop trong `data/adv_index/crops/`), đồng thời lưu vector `768`-chiều (`gemini-embedding-001`) vào bảng PostgreSQL `adv_chunks` (`pgvector`) và log vùng bỏ qua trong `data/adv_index/ingestion_log.json`.
- `build_adv_benchmark.py`: Tạo bộ benchmark 48 câu rời rạc giữa `dev` (18 câu: 3/nhóm × 6 nhóm) và `test` (30 câu: 6 Bảng, 5 Ảnh/sơ đồ, 5 Biểu đồ, 4 Multi-hop, 4 Text, 6 Thiếu bằng chứng), có nguồn trang, `chunk_id` / `crop_path` đối chứng và kiểm chứng 0 trùng trang giữa `dev` và `test`.
- `adv_agent.py`: Hỗ trợ hai chế độ cho ablation trên `dev` và chốt chạy `test`:
  - `text_only`: Hybrid RRF (`pgvector` + BM25) trên text, table Markdown và mô tả ảnh (`max_steps=6`, `budget=16000`).
  - `multimodal`: Agent đa phương thức bổ sung hai vision tool trực tiếp (`inspect_visual_region` nạp ảnh crop PNG, `inspect_page_image` render toàn trang PDF `140 dpi`, `max_steps=8`, `budget=16000`).
- `adv_eval.py`: Kiểm tra đủ 100% nhãn chấm thủ công, không cho phép trace lỗi được tính `pass`, và tổng hợp tỷ lệ đạt `n/N` tổng thể + từng nhóm.

```powershell
# 6. Index tài liệu 150 trang + tạo benchmark nâng cao (dev 18 + test 30)
python src/adv_index.py
python src/build_adv_benchmark.py

# 7. Ablation trên dev (text_only vs multimodal) & chạy pipeline đã chốt trên test
python -u src/adv_agent.py --mode text_only --split dev --model gemini-3.1-flash-lite --output results/adv/dev_text_only_traces.json
python -u src/adv_agent.py --mode multimodal --split dev --model gemini-3.1-flash-lite --output results/adv/dev_multimodal_traces.json
python -u src/adv_agent.py --mode multimodal --split test --model gemini-3.1-flash-lite --output results/adv/test_multimodal_traces.json

# 8. Tổng hợp kết quả chấm thủ công từng câu đối chiếu bản gốc PDF/PNG
python src/adv_eval.py --benchmark data/adv_benchmark.json --traces results/adv/dev_text_only_traces.json --reviews results/adv/dev_text_only_reviews.json --split dev --output results/adv/dev_text_only_summary.json
python src/adv_eval.py --benchmark data/adv_benchmark.json --traces results/adv/dev_multimodal_traces.json --reviews results/adv/dev_multimodal_reviews.json --split dev --output results/adv/dev_multimodal_summary.json
python src/adv_eval.py --benchmark data/adv_benchmark.json --traces results/adv/test_multimodal_traces.json --reviews results/adv/test_multimodal_reviews.json --split test --output results/adv/test_multimodal_summary.json
```
