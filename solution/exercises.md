# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Ở temperature 0.0, bốn lần gọi cho ra gần như cùng một câu trả lời, súc tích và "an toàn". Ở 0.7 câu trả lời vẫn đúng ngữ pháp nhưng đa dạng hơn về cách diễn đạt và chi tiết được chọn. Ở 1.2 bắt đầu xuất hiện câu văn hơi lan man, đôi khi chọn từ bất ngờ. Ở 1.8 mạch lạc giảm rõ rệt — câu có thể lặp từ kỳ lạ, chuyển ý đột ngột, hoặc pha trộn thông tin không liên quan. Điểm "gãy mạch lạc" thường rơi vào khoảng 1.3–1.5 trở lên.*
    
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Trợ lý soạn hợp đồng pháp lý: temperature ~0–0.2, vì cần độ chính xác, nhất quán về thuật ngữ và không được "sáng tạo" thêm điều khoản. Trợ lý viết slogan quảng cáo: temperature ~0.8–1.1, vì mục tiêu là đa dạng ý tưởng, sự bất ngờ và sáng tạo có giá trị hơn tính lặp lại.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Với 20.000 người dùng × 2 lần/ngày × ~500 token output = 20 triệu token đầu ra/ngày (chưa tính input). Với model nhỏ (giá rẻ, ví dụ ~$0.4–0.6/1M token output) chi phí khoảng $8–12/ngày; với model lớn (ví dụ ~$10–15/1M token output) chi phí có thể lên $200–300/ngày — chênh nhau hàng chục lần. Model lớn xứng đáng khi task cần suy luận phức tạp, độ chính xác cao (ví dụ tư vấn y tế/pháp lý), sai sót gây hậu quả lớn. Model nhỏ là lựa chọn đúng cho các tác vụ đơn giản, lặp lại (phân loại, trả lời FAQ) nơi tốc độ và chi phí quan trọng hơn chất lượng biên.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Persona "nhà thơ" cho ra câu trả lời giàu hình ảnh ẩn dụ, giọng văn bay bổng, độ dài trung bình, gần như không có thuật ngữ kỹ thuật. Persona "kỹ sư senior" cho câu trả lời có cấu trúc rõ ràng, dùng thuật ngữ chính xác (weights, gradient descent...), có thể kèm ví dụ code, giọng văn khô và thực dụng hơn. Từ đó thấy system prompt kiểm soát được: giọng điệu, mức độ kỹ thuật/thuật ngữ, cấu trúc trình bày, và loại ví dụ minh họa — nhưng không thay đổi được sự thật cốt lõi (định nghĩa ML vẫn đúng ở cả hai).*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Với tiếng Việt, ước lượng "số từ / 0.75" (dùng cho tiếng Anh) thường sai lệch lớn vì tiếng Việt là ngôn ngữ đơn âm tiết, mỗi "từ" (âm tiết) thường bị tokenizer BPE tách ra thành 1–3 token tùy dấu và độ phổ biến — nhiều hơn tiếng Anh trên cùng số từ. Chênh lệch thực tế thường rơi vào khoảng 30–60% (tiktoken đếm nhiều hơn ước lượng). Nếu dùng công thức thô để dự toán ngân sách, bạn sẽ dự toán thiếu (under-estimate), vì công thức đó được hiệu chỉnh cho tiếng Anh, không phản ánh đúng cách BPE tách tiếng Việt có dấu.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Chatbot văn bản (a) hưởng lợi rõ nhất vì người dùng nhìn thấy chữ xuất hiện dần, giảm cảm giác chờ đợi và tăng trải nghiệm tương tác. Trợ lý giọng nói (b) hưởng lợi vừa phải — cần streaming ở mức để bắt đầu chuyển văn bản sang giọng nói sớm, nhưng không cần hiển thị từng token cho người dùng thấy. Pipeline dịch ngầm ban đêm (c) hầu như không cần streaming, vì không có người theo dõi trực tiếp — chỉ cần kết quả cuối cùng đầy đủ và chính xác, streaming chỉ tăng độ phức tạp code mà không mang lại lợi ích.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Delay cố định khiến hàng nghìn client cùng retry lại đúng cùng một thời điểm, tạo ra một đợt "sóng" request mới làm server tiếp tục quá tải (thundering herd). Exponential backoff giãn thời gian chờ ra tăng dần (1s, 2s, 4s, 8s...) giúp giảm tần suất request theo thời gian, cho server có cơ hội phục hồi. Tuy nhiên nếu tất cả client đều backoff theo đúng cùng một công thức, chúng vẫn có thể đồng bộ retry cùng lúc ở mỗi bước — đây là vấn đề còn sót lại. "Jitter" (thêm độ trễ ngẫu nhiên vào mỗi lần chờ) giải quyết vấn đề này bằng cách làm phân tán thời điểm retry của các client ra, tránh hiện tượng đồng bộ hóa gây ra các đợt tải đột biến lặp lại.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System Prompt: Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt. Hai chỗ mà nếu xóa, hành vi sẽ thay đổi rõ rệt: "trả lời ngắn gọn" — nếu xóa cụm này, trợ lý sẽ có xu hướng trả lời dài dòng, giải thích dài dòng hơn thay vì đi thẳng vào ý chính. "bằng tiếng Việt" — nếu xóa cụm này, AI có thể trả lời bằng tiếng Anh.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Tôi tên là Nguyễn Quốc Anh, tôi bị dị ứng hải sản", sau đó trò chuyện qua 5 lượt khác về chủ đề không liên quan, đến lượt thứ 6 hỏi "Gợi ý cho tôi một quán ăn tối nay" — vì history chỉ giữ 4 lượt cuối, trợ lý đã quên thông tin tôi bị dị ứng hải sản và có thể gợi ý nhà hàng hải sản, gây nguy hiểm/khó chịu cho người dùng. Cách khắc phục: tóm tắt các lượt hội thoại cũ thành vài câu ghi nhớ thông tin quan trọng (tên, sở thích, ràng buộc như dị ứng) và luôn đính kèm bản tóm tắt này vào context, thay vì chỉ giữ nguyên văn 4 lượt gần nhất.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
