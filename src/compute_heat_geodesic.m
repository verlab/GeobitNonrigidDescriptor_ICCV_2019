function heat_flow = compute_heat_geodesic(vertices, faces, keypoints)

    V = vertices;
    F = faces;
    
    heat_flow = zeros(size(V,1), length(keypoints));
    
%     tic;[heat_flow(:,1), ~, ~, ~, ~, pre] = heat_geodesic(V, F, keypoints(1), 1);toc;
%     for k = 2 : length(keypoints)
%         tic;[heat_flow(:,k), ~, ~, ~, ~, ~] = heat_geodesic(V, F, keypoints(k), 1, 'Precomputation', pre);toc;
%     end
    
    tic;[heat_flow(:,1), ~, ~, ~, ~, pre] = heat_geodesic(V, F, keypoints(1), 1);toc;
    %% trying a parallel version
    parfor k = 2 : length(keypoints)
        tic;[heat_flow(:,k), ~, ~, ~, ~, ~] = heat_geodesic(V, F, keypoints(k), 1, 'Precomputation', pre);toc;
    end
end
