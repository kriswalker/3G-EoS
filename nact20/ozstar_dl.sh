#!/bin/bash

for value in 2 3 4
do
  scp -r kwalker@ozstar.swin.edu.au:/fred/oz170/kwalker/projects/3G_EoS/7loudest/outdir_bilby_pipe_${value}/result/summary_plots/config source_${value}
  scp -r kwalker@ozstar.swin.edu.au:/fred/oz170/kwalker/projects/3G_EoS/7loudest/outdir_bilby_pipe_${value}/result/summary_plots/css source_${value}
  scp -r kwalker@ozstar.swin.edu.au:/fred/oz170/kwalker/projects/3G_EoS/7loudest/outdir_bilby_pipe_${value}/result/summary_plots/*html* source_${value}
  scp -r kwalker@ozstar.swin.edu.au:/fred/oz170/kwalker/projects/3G_EoS/7loudest/outdir_bilby_pipe_${value}/result/summary_plots/js source_${value}
  scp -r kwalker@ozstar.swin.edu.au:/fred/oz170/kwalker/projects/3G_EoS/7loudest/outdir_bilby_pipe_${value}/result/summary_plots/plots source_${value}
done
