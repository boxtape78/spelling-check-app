import streamlit as st
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import io
import zipfile
import csv
from collections import Counter

# --- NLTK 설정 (캐싱하여 속도 향상) ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

setup_nltk()

# --- 핵심 로직 ---
def tokenize_text(text):
    return word_tokenize(text)

def is_candidate_word(tok):
    return isinstance(tok, str) and tok.isalpha() and len(tok) > 2 and not tok.isupper()

def count_real_words(text):
    return sum(1 for t in tokenize_text(text) if is_candidate_word(t))

def analyze_and_correct(text, spell):
    tokens = tokenize_text(text)
    candidate_indices = [i for i, t in enumerate(tokens) if is_candidate_word(t)]
    candidate_words = [tokens[i].lower() for i in candidate_indices]
    misspelled_set = spell.unknown(candidate_words)

    corrections = {}
    error_count = 0
    misspelled_list = []
    
    detok = TreebankWordDetokenizer()
    corrected_tokens = list(tokens)

    for idx, lw in zip(candidate_indices, candidate_words):
        if lw in misspelled_set:
            surface = tokens[idx]
            suggestion = spell.correction(lw)
            
            if not suggestion: 
                suggestion = surface
                
            corrections.setdefault(surface, suggestion)
            error_count += 1
            misspelled_list.append(surface)
            
            if surface.istitle():
                final_word = suggestion.capitalize()
            elif surface.isupper():
                final_word = suggestion.upper()
            else:
                final_word = suggestion

            corrected_tokens[idx] = final_word

    corrected_text = detok.detokenize([t if isinstance(t, str) else "" for t in corrected_tokens])
    
    pos_tags = nltk.pos_tag(misspelled_list)
    pos_profile = Counter(tag for word, tag in pos_tags)

    return corrected_text, corrections, error_count, pos_profile

# --- Streamlit 화면 구성 ---
st.title("📝 Spelling Checker & Profiler")
st.markdown("여러 개의 `.txt` 파일을 업로드하면 스펠링을 교정하고 분석 리포트를 제공합니다.")

uploaded_files = st.file_uploader("검사할 텍스트 파일들을 선택하세요", type="txt", accept_multiple_files=True)

if uploaded_files:
    if st.button("스펠링 검사 시작"):
        spell = SpellChecker()
        
        # 결과물을 모을 ZIP 파일 생성 준비
        zip_buffer = io.BytesIO()
        
        error_summary = []
        all_pos_profile = Counter()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # 1. 파일 읽기
                original_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                filename = uploaded_file.name
                
                # 2. 분석 및 교정
                corrected_text, corrections, err_count, pos_profile = analyze_and_correct(original_text, spell)
                all_pos_profile.update(pos_profile)
                
                # 3. 교정된 파일 ZIP에 추가
                zf.writestr(f"corrected_{filename}", corrected_text)
                
                # 4. 요약 정보 저장
                total_words = count_real_words(original_text)
                error_rate = (err_count / total_words * 100) if total_words > 0 else 0
                error_summary.append([filename, total_words, err_count, f"{error_rate:.2f}%"])
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(uploaded_files))

            # 5. CSV 리포트 생성 및 ZIP에 추가
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['Filename', 'Total Words', 'Error Count', 'Error Rate'])
            writer.writerows(error_summary)
            zf.writestr("summary_report.csv", csv_buffer.getvalue())
            
            # 6. 텍스트 리포트(품사 분석 포함) 생성 및 ZIP에 추가
            txt_report = "Total POS Error Profile:\n"
            for tag, count in all_pos_profile.most_common():
                txt_report += f"{tag}: {count}\n"
            zf.writestr("pos_analysis_report.txt", txt_report)

        st.success("✅ 분석 완료!")
        
        # 다운로드 버튼 표시
        st.download_button(
            label="결과
