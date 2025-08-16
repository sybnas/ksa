真实稀有小马拉大车视频_45,全国小马拉大车合集视频大全

   for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int i = 0; i < num; i++)
        {
            int offset = i * N1;
            CHECK(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, 
                cudaMemcpyHostToDevice, streams[i]));
            CHECK(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, 
                cudaMemcpyHostToDevice, streams[i]));

            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1;
            add<<<grid_size, block_size, 0, streams[i]>>>
            (d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, 
                cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
