# 42 subset config resolved values (readiness offline verification)

Verification time: 2026-07-14T10:54:59+09:00
Verification scope: 21 subsets × 2 archs = 42 configs
Eval succeeded: **42/42**  |  Eval errors: 0

- global_batch == 2560: **42/42** ✅
- early_stopping == False: **42/42** ✅

## All config resolved values

| arch | subset | batch | grad_accum | global_batch | LR | max_iters | warmup | early_stop |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bert | mammal_centered | 8 | 80 | 2,560 | 0.0001 | 97,136 | 1,942 | False |
| gpt2 | mammal_centered | 8 | 80 | 2,560 | 0.0006 | 55,021 | 1,100 | False |
| bert | eukaryote_matched_random_seed1 | 8 | 80 | 2,560 | 0.0001 | 96,992 | 1,939 | False |
| gpt2 | eukaryote_matched_random_seed1 | 8 | 80 | 2,560 | 0.0006 | 49,589 | 991 | False |
| bert | eukaryote_matched_random_seed2 | 8 | 80 | 2,560 | 0.0001 | 97,031 | 1,940 | False |
| gpt2 | eukaryote_matched_random_seed2 | 8 | 80 | 2,560 | 0.0006 | 52,697 | 1,053 | False |
| bert | eukaryote_matched_random_seed3 | 8 | 80 | 2,560 | 0.0001 | 97,133 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed3 | 8 | 80 | 2,560 | 0.0006 | 49,131 | 982 | False |
| bert | eukaryote_matched_random_seed4 | 8 | 80 | 2,560 | 0.0001 | 97,128 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed4 | 8 | 80 | 2,560 | 0.0006 | 53,301 | 1,066 | False |
| bert | eukaryote_matched_random_seed5 | 8 | 80 | 2,560 | 0.0001 | 97,149 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed5 | 8 | 80 | 2,560 | 0.0006 | 45,319 | 906 | False |
| bert | eukaryote_matched_random_seed6 | 8 | 80 | 2,560 | 0.0001 | 97,030 | 1,940 | False |
| gpt2 | eukaryote_matched_random_seed6 | 8 | 80 | 2,560 | 0.0006 | 45,837 | 916 | False |
| bert | eukaryote_matched_random_seed7 | 8 | 80 | 2,560 | 0.0001 | 97,148 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed7 | 8 | 80 | 2,560 | 0.0006 | 52,906 | 1,058 | False |
| bert | eukaryote_matched_random_seed8 | 8 | 80 | 2,560 | 0.0001 | 97,121 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed8 | 8 | 80 | 2,560 | 0.0006 | 50,601 | 1,012 | False |
| bert | eukaryote_matched_random_seed9 | 8 | 80 | 2,560 | 0.0001 | 97,116 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed9 | 8 | 80 | 2,560 | 0.0006 | 51,259 | 1,025 | False |
| bert | eukaryote_matched_random_seed10 | 8 | 80 | 2,560 | 0.0001 | 97,119 | 1,942 | False |
| gpt2 | eukaryote_matched_random_seed10 | 8 | 80 | 2,560 | 0.0006 | 49,105 | 982 | False |
| bert | global_random_seed1 | 8 | 80 | 2,560 | 0.0001 | 97,149 | 1,942 | False |
| gpt2 | global_random_seed1 | 8 | 80 | 2,560 | 0.0006 | 47,559 | 951 | False |
| bert | global_random_seed2 | 8 | 80 | 2,560 | 0.0001 | 97,051 | 1,941 | False |
| gpt2 | global_random_seed2 | 8 | 80 | 2,560 | 0.0006 | 52,351 | 1,047 | False |
| bert | global_random_seed3 | 8 | 80 | 2,560 | 0.0001 | 97,149 | 1,942 | False |
| gpt2 | global_random_seed3 | 8 | 80 | 2,560 | 0.0006 | 46,501 | 930 | False |
| bert | global_random_seed4 | 8 | 80 | 2,560 | 0.0001 | 97,149 | 1,942 | False |
| gpt2 | global_random_seed4 | 8 | 80 | 2,560 | 0.0006 | 51,826 | 1,036 | False |
| bert | global_random_seed5 | 8 | 80 | 2,560 | 0.0001 | 97,143 | 1,942 | False |
| gpt2 | global_random_seed5 | 8 | 80 | 2,560 | 0.0006 | 52,027 | 1,040 | False |
| bert | global_random_seed6 | 8 | 80 | 2,560 | 0.0001 | 77,662 | 1,553 | False |
| gpt2 | global_random_seed6 | 8 | 80 | 2,560 | 0.0006 | 36,848 | 736 | False |
| bert | global_random_seed7 | 8 | 80 | 2,560 | 0.0001 | 97,149 | 1,942 | False |
| gpt2 | global_random_seed7 | 8 | 80 | 2,560 | 0.0006 | 51,042 | 1,020 | False |
| bert | global_random_seed8 | 8 | 80 | 2,560 | 0.0001 | 97,086 | 1,941 | False |
| gpt2 | global_random_seed8 | 8 | 80 | 2,560 | 0.0006 | 49,675 | 993 | False |
| bert | global_random_seed9 | 8 | 80 | 2,560 | 0.0001 | 96,954 | 1,939 | False |
| gpt2 | global_random_seed9 | 8 | 80 | 2,560 | 0.0006 | 49,990 | 999 | False |
| bert | global_random_seed10 | 8 | 80 | 2,560 | 0.0001 | 96,671 | 1,933 | False |
| gpt2 | global_random_seed10 | 8 | 80 | 2,560 | 0.0006 | 49,064 | 981 | False |
