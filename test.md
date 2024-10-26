to check the time cost of 3DGS model (add `--test_no_render_laststep` to see the time cost of purely passing memories in step 4)
``` bash
python fps_test_amr_1080p_foveated.py -m output/01703a7c-c
```

to check the time cost of eye-tracking model (add `--foveal_layer_timer` to see the time cost by layers)

```bash
python -i track.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100 
```


to run the two models in parallel

```bash
 mpirun -np 2 python parallel_track_render.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/01703a7c-c
```

