# Speaker Notes for Proposal Deck

## 開場

先講醫院，不要先講演算法。老師提醒大家一開始不會知道我們在做什麼，所以先用共同直覺：不同醫院速度不同、資料不能集中、同步會等慢節點。

## Methodology

重點是 CAA-v2 不是 FedBuff 改名。FedBuff 只 buffer async updates；CAA-v2 在 buffer 內加入 logical staleness、direction agreement、server trajectory EMA、delta clipping、client fairness credit、adaptive alpha。全部只使用 logical version、delta、client id、contribution count，不用 global physical clock。

## Experiment Results

先講公平比較：async events = sync rounds × clients。然後講 9 datasets × 6 methods × 3 seeds。誠實說 CAA-v2 不是 universal winner，但 overall mean best/final accuracy 高於 Sync / Naive / FedBuff，且比 Naive 和 CAA-v1 穩。

## Challenges

把挑戰講成 distributed systems 問題：no global clock、stale update、conflicting update、fast-client domination、privacy tension、evaluation fairness。這會比單純講 accuracy 更符合課程精神。
