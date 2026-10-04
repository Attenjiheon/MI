# P10 Update block 3 준비 검증

2026-10-03. **준비 검증 통과. P10 실제 학습·평가 미실행.**

사용자 자원은 T4, 잔여 24.59 CU다. phase 권장 순서에 따라 Update block 3 한 층의 SAE/TC 12 runs를 별도 실험 ID로 동결했다.
실제 CU/hour는 확인되지 않았으며 시간 보장은 하지 않는다. 추가 층·length·m→m SAE·Update 인과 개입은 이번 범위에서 생략한다.
선택 근거와 전달·재개 절차는 [P10 상태](../../P10_STATUS.md), 정확한 파일 hash와 명령은 [verification.json](verification.json)을 따른다.

## 확인한 증빙

- P9 완료 판정의 7개 연결 hash가 실제 파일과 일치한다. 원본 반환 ZIP 594개 member hash와 16개 run identity/통과 기록을 재확인했다. 기존 학습·본평가를 재실행하지 않았다.
- 동결 corpus manifest의 예약 Update 위치 및 interpretation 원본 hash, frozen LM 3개와 P5가 확인한 초기 checkpoint 3개의 hash를 대조했다.
- 예약 50,000/10,000/20,000개 위치를 독립 토큰 parser로 replay했다. 3,138/631/1,255개 sequence의 canonical hash는 split 간 겹치지 않는다. 초기화 SET은 포함되지 않는다.
- 8개 기본 라벨과 각 4개 변수/연산 전이, 총 16개 과제의 train/val support가 기준을 만족한다. 결측 src·입력 관계는 0으로 채우지 않는다. Test 성능이나 checkpoint/feature 선택을 수행하지 않았다.
- ZIP 91개 member hash 일치. 번들만 별도 임시 디렉터리에 풀어 코드·입력 검증과 8개 테스트를 실행했다. 테스트 8개 통과. 노트북 schema 및 코드 셀 8개 구문 검증 통과.
- CPU Update smoke: h/u/m 위치 및 수식, 미래 토큰 불변성, identity patch, SAE/TC 두 k의 bitwise resume 검증 통과. GPU smoke는 노트북에서 새로 실행한다.

## 전달물과 한계

입력 ZIP은 434,917,737 bytes이며 SHA-256은 `58be7323e59474e659cf350034900c6e3eaf7692fd884462c3c4cddbc8bdbdf9`다.
학습용 첫 전달물이며 의미·fidelity·전이 실행은 선택 checkpoint의 반환 감사 후 진행한다. 평가 규칙과 candidate subset/RNG는 현재 계약에 먼저 고정했다.
CPU 준비를 CUDA 실행으로 간주하지 않는다. 12 runs/60k updates/30.72M draws와 252 checkpoint/240 validation 값은 예정량이다.
P10 및 전체 실험은 미완료이며 완료 체크박스나 과거 동결 manifest를 변경하지 않았다.
