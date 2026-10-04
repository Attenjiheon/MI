# P10 Update 평가 전달물 준비

2026-10-03. **준비 검증 통과, 실제 평가 0/12.**

기존 Update 평가 규칙과 학습 반환 감사의 선택 모델 12개를 별도 평가 계약에 연결했다. Test 결과를 보고 규칙이나 checkpoint를 고르지 않았다.

- T4 단계: 검증된 학습 train/val cache 재사용, trained test 및 초기 LM train/val/test cache 추출, 동결 dictionary latent·fidelity 계산.
- CPU 단계: 1,248개 fitting 과제. 공유 full/초기 LM/token-position/shuffled 기준선 480개, 좌표/random 전체·128후보 384개, latent 전체·128후보 384개. 단일/≤4를 포함한 의미 보고는 2,016개다.
- 변수 A/B/C→D 및 AND/OR→XOR는 source-domain train/val에서 전처리·ranking·lambda·m·threshold를 다시 fit한다. 모든 선택 파일을 고정한 뒤 test 보고를 시작한다.
- 각 지표에 sequence cluster bootstrap 1,000회와 유효 draw 수를 기록한다. Full probe와의 balanced accuracy·macro F1 차이는 동일 sequence draw를 사용한 paired CI다.
- 하나의 과제 또는 cache chunk를 완전 저장한 뒤 재개한다. 이미 기록된 optimizer 실패를 건너뛰어 성공 처리하지 않는다. 부정적 결과 및 support 부족을 보존한다.

## 검증

번들 119개 member hash와 checksum, 두 노트북 schema·코드 구문 검증 통과. 실제 번들을 임시 디렉터리에 풀어 독립 실행한 테스트 5개가 통과했다.
선택된 실제 checkpoint 12개를 각각 synthetic activation 17행/split에 적용하여 latent 및 MSE를 별도 F.linear/TopK 식과 대조했다. 이는 debug fixture이며 실제 LM test inference나 실제 fidelity 결과가 아니다.

입력 ZIP은 469,480,370 bytes, SHA-256 `69bb4191321ed5228cdb5a5dd0e90f9cbeaf1e506aedd1a85a87c9ef87b22e1e`다.
정확한 파일 hash와 명령은 [verification.json](verification.json)을 따른다. CUDA 실제 실행, 본평가, 원본 cache 재현 감사는 아직 남아 있다.
