## Alive-layer sensitivity (bar, ball exact, clean_tracks True, who who_v1) -- one error kind at a time on the TRUE alive mask

```
              case  spellF1  d_vs_oracle    se  macroF1  alive_dead_f1  n
            oracle    0.670        0.000 0.000    0.938          1.000 11
 restart_late 0.2s    0.666       -0.004 0.001    0.935          0.997 11
 restart_late 0.5s    0.652       -0.014 0.002    0.930          0.993 11
   restart_late 1s    0.627       -0.031 0.004    0.921          0.986 11
   restart_late 2s    0.602       -0.053 0.006    0.906          0.972 11
restart_early 0.5s    0.659       -0.013 0.001    0.930          0.993 11
  restart_early 1s    0.643       -0.027 0.001    0.922          0.987 11
    stop_late 0.5s    0.654       -0.018 0.001    0.930          0.993 11
      stop_late 1s    0.632       -0.039 0.002    0.921          0.986 11
      stop_late 2s    0.603       -0.068 0.003    0.905          0.973 11
   stop_early 0.5s    0.660       -0.012 0.002    0.933          0.993 11
     stop_early 1s    0.648       -0.022 0.002    0.926          0.986 11
          lag 0.5s    0.633       -0.034 0.003    0.922          0.987 11
            lag 1s    0.588       -0.072 0.005    0.904          0.973 11
            lag 2s    0.536       -0.120 0.007    0.873          0.946 11
     drop_short 3s    0.668       -0.003 0.001    0.937          0.999 11
     drop_short 5s    0.659       -0.011 0.001    0.933          0.995 11
     drop_short 8s    0.639       -0.029 0.003    0.919          0.984 11
    drop_short 12s    0.609       -0.056 0.004    0.892          0.964 11
   blips 1s x10/90    0.660       -0.012 0.002    0.937          0.999 11
   blips 2s x10/90    0.653       -0.016 0.002    0.933          0.995 11
   blips 4s x10/90    0.651       -0.017 0.002    0.929          0.992 11
   blips 2s x30/90    0.632       -0.038 0.003    0.927          0.991 11
     dead_v1 (HMM)    0.550       -0.116 0.003    0.860          0.933 11
```
