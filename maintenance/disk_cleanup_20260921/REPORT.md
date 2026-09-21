# 디스크 정리 — 2026-09-21

사용자가 디스크 부족을 이유로 불필요한 원본 삭제와 후속 단계에 필요한 기록의 압축 보존을 명시적으로 요청했다. 이 요청은 과거 원본을 모두 유지하던 AGENTS.md §4·§11 및 README의 보존 원칙에 대한 **이번 목록에 한정된 예외**다. 실험 선택·gate·상태는 변경하지 않았다.

## 삭제 및 기록

- 완료된 로컬 smoke/resume 검증용 `.pt` 101개: 약 4.34 GiB의 논리 크기. 일부는 hardlink이므로 실제 회수 크기는 더 작다. 기존 JSON 결과·환경 lock·resume equivalence·실패/성공 기록은 원래 경로에 유지했다.
- 이전 전달 ZIP 9개: 약 4.75 GiB. 각 ZIP의 전체 SHA-256, member 경로·크기·CRC, 내부의 작은 코드·설정·manifest·노트북을 `bundle_records/`에 압축 보관했다. 원본 ZIP과 동일한 압축 바이너리를 복구할 수 있다는 뜻은 아니다.
- 거부된 최초 v1.4 코퍼스 payload 318개: 약 1.74 GiB. `data/language_v1_4/rebuild_01/`은 제외했다. 거부본 manifest·통계·검증 JSON·vocab·provenance는 원래 경로에 유지했다. 거부 사유는 `experiment_v1_4/corpus_rebuild.json`을 따른다. 삭제한 원본은 더 이상 로컬에서 재감사할 수 없으며, hash 기록은 데이터의 대체물이 아니다.
- Git LFS 임시 파일: 파일을 연 프로세스가 없고 120초 이상 변경되지 않았음을 확인한 7개만 삭제했다. 총량은 `result.json`에 있다.
- Python/pytest/Finder 캐시를 정리했다.

`plan.json`은 사전 목록, `deleted.jsonl`은 각 원본을 삭제하기 **전에 fsync한** 경로·크기·SHA-256·inode·삭제 사유다. `bundle_records/`는 번들별 축약 기록이다. `execution.log`, `lfs_tmp_deleted.jsonl`, `cache_deleted.json` 및 `tracked_deletions.txt`도 보존한다. `execute.py`는 일회성 스크립트이며 다시 실행하지 않는다.

## 후속 실험 보존 및 검증

- 활성 v1.4 rebuild_01의 330개 corpus 파일과 동결 config 전체 검증 통과.
- seed 0 completion의 모든 참조 증빙 및 P4 replication authorization 검증 통과.
- P4 입력 ZIP, seed 1·2 노트북, P4 빌더의 입력인 P3 r3 ZIP의 SHA-256 일치.
- seed 0 반환 원본 ZIP과 내부 selected/last/init checkpoint를 포함한 원본 증빙은 유지. ZIP 전체 SHA-256 일치.
- 모든 버전의 소스·설계·config·결과 보고서와 기존 실패 결과를 유지. 과거 버전의 반환 증빙 ZIP 및 활성 이전 corpus도 유지했다.
- 모델 실행·gate/test 재평가 없음. P4 실제 GPU 실행은 여전히 미실행.

검증 증거는 `preserved_verification.json`이다. 옛 문서의 원본 보존 문구와 checksum은 **당시의 사실**이며 현재 가용성은 이 문서와 삭제 목록으로 보완한다. 동결 문서·manifest·실험 결과를 수정하지 않았다. 삭제한 디버그 checkpoint로 과거 resume 검증을 재실행하려면 새 출력 경로에서 smoke를 재생성해야 한다.

## Git 및 남은 용량

Git 이력·객체·worktree는 변경하지 않았다. 원격 복사본 검증을 요청한 LFS prune dry-run은 기존 `fatal: bad object main 2` 오류로 실패했다. 이후 원본과 같은 LFS 캐시만 삭제하는 제안도 원격 복사본 미확인을 이유로 자동 승인 검토에서 거절되어 실행하지 않았다. `.git/lfs/objects`는 그대로 유지한다.

시작 시 P4 준비 작업에 해당하는 다수의 미커밋 변경이 존재했다. 이 정리는 phase 완료 작업이 아니므로 해당 변경을 섞어 커밋·push하지 않았다. `git_head_before.txt`와 추적 파일 삭제 목록을 남겼다. Git의 일반 status/diff는 큰 LFS 파일의 clean filter를 실행하여 임시 공간을 소비할 수 있다.

실제 파일 삭제량과 디스크 여유 공간은 `result.json`에 기록했다. 파일시스템 snapshot·동시 작업 때문에 삭제 바이트 합계와 df의 변화는 일치하지 않을 수 있다.
