import os
import textwrap
import streamlit as st
from typing import Optional

try:
    # New-style OpenAI SDK (>=1.0)
    from openai import OpenAI  # type: ignore
    _SDK_STYLE = "new"
except Exception:
    # Fallback for legacy SDK
    import openai  # type: ignore
    _SDK_STYLE = "legacy"

# ---------------------------
# Helpers
# ---------------------------

def _get_api_key() -> Optional[str]:
    # 1) Streamlit Cloud secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # 2) Environment variable (local dev)
    return os.getenv("OPENAI_API_KEY")


def _get_client():
    api_key = _get_api_key()
    if not api_key:
        st.stop()
    if _SDK_STYLE == "new":
        return OpenAI(api_key=api_key)
    else:
        openai.api_key = api_key
        return None  # not used


def generate_novel(
    protagonist1: str,
    protagonist2: str,
    genre: str,
    premise: str,
    language: str,
    target_words: int = 1200,
    temperature: float = 0.9,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI to generate a short novel based on inputs."""
    style_guide = textwrap.dedent(
        f"""
        Write a complete short story in {language}.
        Requirements:
        - Genre: {genre}
        - Main characters: {protagonist1} and {protagonist2}
        - Start with a strong hook in the first 2–3 sentences.
        - Maintain clear scene breaks and a satisfying arc (setup → escalation → climax → resolution).
        - Show, don't tell. Keep dialogue natural.
        - Keep pacing tight and avoid filler.
        - Tone and diction should match the genre.
        - Target length: ~{target_words} words (±15%).
        - End with a resonant final line (no meta commentary).

        Backstory / seed premise:
        {premise.strip()}
        """
    ).strip()

    client = _get_client()

    # New SDK path
    if _SDK_STYLE == "new":
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=4096,  # enough for ~3k words; safety is enforced by target_words
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an award-winning novelist. You craft vivid scenes, believable dialogue, and tight plots."
                        ),
                    },
                    {
                        "role": "user",
                        "content": style_guide,
                    },
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    # Legacy SDK path
    else:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                max_tokens=4096,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an award-winning novelist. You craft vivid scenes, believable dialogue, and tight plots."
                        ),
                    },
                    {
                        "role": "user",
                        "content": style_guide,
                    },
                ],
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI 소설 생성기", page_icon="📖", layout="wide")

st.title("📖 AI 소설 생성기")
st.caption("주인공 이름과 장르, 간략한 스토리를 입력하면 단편 소설을 생성합니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    default_lang = st.selectbox("출력 언어", ["한국어", "English"], index=0)
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    target_words = st.slider("목표 길이(단어)", min_value=400, max_value=3000, value=1200, step=100)
    temperature = st.slider("창의성(temperature)", min_value=0.0, max_value=1.5, value=0.9, step=0.1)
    st.markdown(
        """
        **바라바라바라밤**
        - 어서와 친구들 !!.
        - 내가 웹페이지를 만들었어!.
        """
    )

col1, col2 = st.columns(2)
with col1:
    protagonist1 = st.text_input("주인공 이름 1", placeholder="예: 지우")
with col2:
    protagonist2 = st.text_input("주인공 이름 2", placeholder="예: 민서")

genre = st.selectbox(
    "장르",
    [
        "로맨스",
        "미스터리",
        "스릴러",
        "판타지",
        "SF",
        "호러",
        "성장소설",
        "휴먼드라마",
        "코미디",
        "사극/역사",
    ],
)

premise = st.text_area(
    "간략한 스토리(프롬프트)",
    placeholder=(
        "예: 오래된 북촌 골목의 소품 가게를 지키는 지우와, 시간을 되감는 필름을 찾는 민서.\n"
        "한여름 밤 폭우 속, 그들이 맞이하는 선택의 순간."
    ),
    height=160,
)

cta = st.button("소설 생성하기 🚀", type="primary")

if cta:
    api_key = _get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. 사이드바 안내를 참고해 키를 등록하세요.")
        st.stop()

    # Basic validation
    if not protagonist1 or not protagonist2 or not premise:
        st.warning("주인공 이름 2개와 스토리 프롬프트를 모두 입력해주세요.")
        st.stop()

    with st.spinner("소설을 집필 중입니다…"):
        try:
            story = generate_novel(
                protagonist1=protagonist1.strip(),
                protagonist2=protagonist2.strip(),
                genre=genre,
                premise=premise,
                language=default_lang,
                target_words=int(target_words),
                temperature=float(temperature),
                model=model,
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.success("완료! 아래에서 결과를 확인하세요.")

    # Display
    st.subheader("📝 생성된 소설")
    st.download_button(
        label="소설 텍스트 다운로드 (.txt)",
        data=story.encode("utf-8"),
        file_name=f"novel_{protagonist1}_{protagonist2}.txt",
    )

    st.markdown("---")
    st.markdown(story)

# Footer
st.markdown("\n\n— 흐흐흐흐흐흐흐흐흐흐아각각!!!")
