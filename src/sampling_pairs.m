%% sampling on a polar coordinate system

clear all
close all

n_pairs = 512;
max_distance = 15;

% cartesian coordinates
pairs = zeros(n_pairs, 4);
%x
% radius (distances) and` angles
r_and_theta = zeros(n_pairs, 4);

for n = 1 : n_pairs
    
    %wrong method
    %r1(n)=R*rand(1,1);
    %theta1(n)=2*pi*rand(1,1);
    
    % right method
    % pdf_r(r)=(2/R^2) * r
    % cumulative pdf_r is F_r = (2/R^2)* (r^2)/2
    % inverse cumulative pdf is r = R*sqrt(F_r)
    % so we genera  te the correct r as

    for p = 1 : 2
        
        %wrong method concentration on the origin
        %r1(n)=R*rand(1,1);
        %theta1(n)=2*pi*rand(1,1);

        % [r1 theta1 r2 theta2]
        % right way to smapling randomly on a circle
        r_and_theta(n, 1+(p-1)*2) = cast(max_distance * sqrt(rand(1,1)), 'uint8');
        
        % concentration on the center of the circle
        %r_and_theta(n, 1+(p-1)*2) = cast(max_distance * rand(1,1), 'uint8');
        
        
        % and theta as before:
        r_and_theta(n, 2+(p-1)*2) = 2 * pi * rand(1,1);
        
        % convert to cartesian for drawing
        r = r_and_theta(n, 1+(p-1)*2);
        theta = r_and_theta(n, 2+(p-1)*2);
        
        pairs(n, 1+(p-1)*2) = r * cos(theta); % x1, y1
        pairs(n, 2+(p-1)*2) = r * sin(theta); % x2, y2
    end
end

% Saving

%save('test_pairs_128.txt', 'r_and_theta','-ascii');

%subplot(1,2,2)
plot(pairs(:,1:2)', pairs(:,3:4)', 'r-o')

axis([-1.1*max_distance 1.1*max_distance -1.1*max_distance 1.1*max_distance])
axis square

