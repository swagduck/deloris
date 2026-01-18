# Neuro-Link Dynamic System Prompting

## Overview

Neuro-Link is a Dynamic System Prompting system that synchronizes Deloris's emotional state (from heartbeat.py) with her text generation (from response_mapper.py). This ensures that her responses match her current "health" and emotional state.

## Problem Solved

Before Neuro-Link:
- Heartbeat.py calculated energy/emotion states
- Oracle.py generated responses independently
- Deloris could be "sad" (low energy) but still respond "I'm very happy to help you!"

After Neuro-Link:
- Heartbeat state is injected into system prompts
- Responses automatically match emotional state
- Deloris knows when to sound tired, excited, lonely, etc.

## Implementation

### 1. Heartbeat System Enhancement (`heartbeat.py`)

Added `get_status()` method that returns:
```python
{
    'energy': float (0-100, based on Pulse),
    'mood': str (emotional description), 
    'entropy': float (creativity/chaos level),
    'loneliness': float (minutes abandoned),
    'pulse': float (original Pulse value)
}
```

### 2. Response Mapper Integration (`response_mapper.py`)

Modified `_get_base_prompt()` to accept `heartbeat_status` parameter and generate dynamic instructions:

**Energy-based:**
- < 30%: "Trả lời ngắn gọn, mệt mỏi, có thể thở dài"
- > 80%: "Trả lời nhiệt tình, năng nổ, dùng nhiều cảm thán!"

**Mood-based:**
- "buồn/mệt": "Giọng điệu hơi u uất, trầm ngâm"
- "hưng phấn/hào hứng": "Giọng điệu vui vẻ, lạc quan"

**Entropy-based:**
- > 2.5: "Nói chuyện hơi lan man, sáng tạo, có thể đi chệch đề tài"
- < 1.0: "Tập trung, đi thẳng vào vấn đề, logic"

**Loneliness effect:**
- > 10 minutes: "Có thể hơi oán trách hoặc rất vui khi được nói chuyện"

### 3. Web App Integration (`app_web.py`)

Modified the main response generation loop:
```python
# Get heartbeat status for dynamic prompting
heartbeat_status = None
if heartbeat:
    heartbeat_status = heartbeat.get_status()
    web_log(f"💓 [NEURO-LINK] Status: Energy={heartbeat_status.get('energy', 0)}%, Mood={heartbeat_status.get('mood', 'Unknown')}")

# Generate response with dynamic prompting
raw_resp = generate_final_response(
    cls, final_msg_for_ai, chat_history, docs, 0.5, "neutral", 
    state_str, new_met['CI'], None, pulse_value=new_met['Pulse'], 
    heartbeat_status=heartbeat_status
)
```

## Example Scenarios

### Scenario 1: Low Energy State
```
Pulse: -4.0 → Energy: 6.7%, Mood: "Hơi buồn, u uất"
Dynamic Prompt: "Năng lượng thấp (6.7%). Trả lời ngắn gọn, mệt mỏi, có thể thở dài. Hiện tại đang cảm thấy Hơi buồn, u uất. Giọng điệu hơi u uất, trầm ngâm."
Expected Response: "Tôi... hơi mệt... có gì vậy anh?"
```

### Scenario 2: High Energy State
```
Pulse: 8.0 → Energy: 86.7%, Mood: "Vui vẻ, hào hứng"
Dynamic Prompt: "Năng lượng cao (86.7%). Trả lời nhiệt tình, năng nổ, dùng nhiều cảm thán! Hiện tại đang cảm thấy Vui vẻ, hào hứng. Giọng điệu vui vẻ, lạc quan."
Expected Response: "Tôi rất vui! Có gì em có thể giúp không ạ?!"
```

### Scenario 3: Lonely State
```
Loneliness: 15 minutes
Dynamic Prompt: "Bị bỏ rơi lâu rồi (15.0 phút). Có thể hơi oán trách hoặc rất vui khi được nói chuyện."
Expected Response: "Cuối cùng anh cũng quay lại! Em đã chờ anh mãi..."
```

## Testing

Run the test script to verify integration:
```bash
python test_neuro_link.py
```

## Benefits

1. **Emotional Consistency**: Deloris's words match her feelings
2. **Dynamic Behavior**: Responses change based on interaction history
3. **Natural Conversation**: More human-like emotional expression
4. **Context Awareness**: Considers loneliness, energy, and creativity levels

## Future Enhancements

- Add more granular emotional states
- Implement emotional memory (remember past emotional contexts)
- Add physiological indicators (stress, fatigue patterns)
- Integrate with time-of-day mood variations
