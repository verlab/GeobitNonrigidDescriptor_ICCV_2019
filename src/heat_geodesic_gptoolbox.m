clc;
clear;
close all;

%% setup
% toolboxes
gp_subdirs = split(genpath('/Users/erickson/Dropbox/ufmg/sabatico/nonrigid_desc_project/codes/toolboxes/gptoolbox-master'),':');
addpath(strjoin(gp_subdirs(~contains(gp_subdirs,'.git')),':'));

toolboxgraph_subdirs = split(genpath('/Users/erickson/Dropbox/ufmg/sabatico/nonrigid_desc_project/codes/toolboxes/toolbox_graph_and_meshes/toolbox_graph'),':');
addpath(strjoin(gp_subdirs(~contains(toolboxgraph_subdirs,'.git')),':'));

% 3d models
%addpath('/Users/erickson/Dropbox/ufmg/sabatico/nonrigid_desc_project/framework-code/models');
addpath('C:\Users\Erickson\Dropbox\ufmg\sabatico\nonrigid_desc_project\framework-nonrigid\data');

%clouds = {'cloud_1.pcd.ply'; 'cloud_2.pcd.ply'};
clouds = {'cloud_1_noorigin_pcd.ply'; 'cloud_2.pcd.ply'};

%% select test
rand_selection = 0;
cloud = 1;
Nkp = 3;

%% load data

disp(clouds{cloud});
[V, F] = readPLY(clouds{cloud});
disp('model loaded');

size(V)
size(F)

all_vertices_indices = 1:length(V);
triangles_vertices = union(union(F(:,1), F(:,2)), F(:,3));
connected_vertices = ismember(all_vertices_indices, triangles_vertices);

%% select keypoints

keypoints = [104053.0 98285.0 100858.0];
if (rand_selection)
%select randomly Nkp keypoints avoiding (0,0,0)
    invalid_point = V(:,1) == 0 & V(:,2) == 0 & V(:,3) == 0;
    idx = find(~invalid_point);
    
    keypoints = idx(randi(length(idx), Nkp, 1));
end

disp('Selected keypoints');
V(keypoints,:)

%% compute heat flow (approximation of the geodesic distance)

heat_flow = compute_heat_geodesic(V, F, keypoints);

%% comparsion with cpp call
%cpp_heat = load('C:\Users\Erickson\Dropbox\ufmg\sabatico\nonrigid_desc_project\framework-nonrigid\code\compute_descriptor\build\Debug\heat_flow.txt');
%img_cpp_heat = reshape(cpp_heat(1,:), [640,480]);

%p = 30;
%DispFunc = @(phi)cos(2*pi*p*phi);
%img_cpp_heat = DispFunc(img_cpp_heat);

%figure;imshow(img_cpp_heat');
%colormap parula(256);

%hold on;plot(mod(keypoints(1),640), keypoints(1)/640, 'ro');

%keyboard;

for i = 1 : size(heat_flow,2)
        
    D = heat_flow(:,i);
    
    % handling NaN (possible due to vertices on (0,0,0))
    D( isnan(D) ) = max( D );
    
    p = 30;
    DispFunc = @(phi)cos(2*pi*p*phi);
    options.face_vertex_color = DispFunc(D);

    figure;plot_mesh(V, F, options);
    
    axis('tight');
    colormap parula(256);
    
    hold on;
    plot3(V(keypoints(i),1), V(keypoints(i),2), V(keypoints(i),3), 'r.', 'MarkerSize', 25);
    
    interesting_radius = D < 0.05*max(D);
    plot3(  V(interesting_radius,1), ...
            V(interesting_radius,2), ...
            V(interesting_radius,3), 'y.', 'MarkerSize', 10);
        
    figure;plot(sort(D(interesting_radius)), 'r.');
    hist_dist = histcounts(D(interesting_radius), 100);
    figure;histogram(hist_dist);
end