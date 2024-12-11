
# /output/e26eae8e-f: ~/gaussian_splatting_with_eye_tracking/db/playroom
# /output/29554d64-8: ~/gaussian_splatting_with_eye_tracking/db/drjohnson
# /output/36cf0258-6: ~/gaussian_splatting_with_eye_tracking/tandt/truck
# /output/06008696-3: ~/gaussian_splatting_with_eye_tracking/tandt/train

# Process for train
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_track_render_timing.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/06008696-3/ --foveal_cpu  > log.txt 2>&1 
mv fovealnet_timing_data.json fovealnet_timing_data_cpu_timing.json
mv timing_data.json timing_data_cpu_timing.json
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_load_balance.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/06008696-3/ --foveal_cpu  > log.txt 2>&1 
python parallel_track_render_plot.py --first_image_separate
mv rendering_time_per_step.png rendering_time_per_step_train.png
mv fovealnet_time_per_layer.png fovealnet_time_per_layer_train.png
mv fovealnet_timing_data.json fovealnet_timing_data_train.json
mv timing_data.json timing_data_train.json
mv fovealnet_timing_data_first_image.json fovealnet_timing_data_first_image_train.json
mv fovealnet_timing_data_cpu_timing.json fovealnet_timing_data_cpu_timing_train.json
mv timing_data_cpu_timing.json timing_data_cpu_timing_train.json

# Process for truck
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_track_render_timing.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/36cf0258-6/ --foveal_cpu  > log.txt 2>&1 
mv fovealnet_timing_data.json fovealnet_timing_data_cpu_timing.json
mv timing_data.json timing_data_cpu_timing.json
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_load_balance.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/36cf0258-6/ --foveal_cpu  > log.txt 2>&1 
python parallel_track_render_plot.py --first_image_separate
mv rendering_time_per_step.png rendering_time_per_step_truck.png
mv fovealnet_time_per_layer.png fovealnet_time_per_layer_truck.png
mv fovealnet_timing_data.json fovealnet_timing_data_truck.json
mv timing_data.json timing_data_truck.json
mv fovealnet_timing_data_first_image.json fovealnet_timing_data_first_image_truck.json
mv fovealnet_timing_data_cpu_timing.json fovealnet_timing_data_cpu_timing_truck.json
mv timing_data_cpu_timing.json timing_data_cpu_timing_truck.json

# Process for drjohnson
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_track_render_timing.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/29554d64-8/ --foveal_cpu  > log.txt 2>&1 
mv fovealnet_timing_data.json fovealnet_timing_data_cpu_timing.json
mv timing_data.json timing_data_cpu_timing.json
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_load_balance.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/29554d64-8/ --foveal_cpu  > log.txt 2>&1 
python parallel_track_render_plot.py --first_image_separate
mv rendering_time_per_step.png rendering_time_per_step_drjohnson.png
mv fovealnet_time_per_layer.png fovealnet_time_per_layer_drjohnson.png
mv fovealnet_timing_data.json fovealnet_timing_data_drjohnson.json
mv timing_data.json timing_data_drjohnson.json
mv fovealnet_timing_data_first_image.json fovealnet_timing_data_first_image_drjohnson.json
mv fovealnet_timing_data_cpu_timing.json fovealnet_timing_data_cpu_timing_drjohnson.json
mv timing_data_cpu_timing.json timing_data_cpu_timing_drjohnson.json

# Process for playroom
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_track_render_timing.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/e26eae8e-f/ --foveal_cpu  > log.txt 2>&1 
mv fovealnet_timing_data.json fovealnet_timing_data_cpu_timing.json
mv timing_data.json timing_data_cpu_timing.json
mpirun -np 2 --bind-to core --rankfile rankfile.txt  python parallel_load_balance.py --foveal_layer_timer --eye_image_sequence_id_start 0 --eye_image_sequence_id_end 100  -m output/e26eae8e-f/ --foveal_cpu  > log.txt 2>&1 
python parallel_track_render_plot.py --first_image_separate
mv rendering_time_per_step.png rendering_time_per_step_playroom.png
mv fovealnet_time_per_layer.png fovealnet_time_per_layer_playroom.png
mv fovealnet_timing_data.json fovealnet_timing_data_playroom.json
mv timing_data.json timing_data_playroom.json
mv fovealnet_timing_data_first_image.json fovealnet_timing_data_first_image_playroom.json
mv fovealnet_timing_data_cpu_timing.json fovealnet_timing_data_cpu_timing_playroom.json
mv timing_data_cpu_timing.json timing_data_cpu_timing_playroom.json
