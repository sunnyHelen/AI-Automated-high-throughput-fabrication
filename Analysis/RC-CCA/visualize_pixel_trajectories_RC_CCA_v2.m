%% Pixel Trajectory Visualization Tool - Version 2
% Creates cumulative intensity maps showing where signals appeared over time
% - Gray color: Green channel (vessel) trajectory with accumulated opacity
% - Blue color: Red channel (platelet) trajectory with accumulated opacity
%
% NEW: Resizes all videos to 512x512, then crops to 360x360 centered region
%
% Output: One figure per video showing temporal signal patterns
%
% Author: Trajectory Visualization Tool
% Date: November 2025

clear all; close all; clc;

%% Configuration
config.videoFolder = './videos/'; % Folder containing all .mp4 videos
config.outputFolder = './trajectory_figures/'; % Folder for saving trajectory figures

% Create output directory if it doesn't exist
if ~exist(config.outputFolder, 'dir')
    mkdir(config.outputFolder);
end

% Crop parameters (360x360 crop, adjustable center position)
config.analysisCropSize = 360;
config.cropCenterX = 300;  % X center of crop region (in 512x512 coordinates)
config.cropCenterY = 240;  % Y center of crop region (in 512x512 coordinates)

% Processing parameters
config.greenThreshold = 50;  % Minimum intensity to count as green signal (0-255)
config.redThreshold = 10;    % Minimum intensity to count as red signal (0-255)
config.redPurityRatio = 1.5; % Red must be this much stronger than green to count as pure red

%% Step 1: Find all videos
fprintf('=== Finding Videos ===\n');
videoFiles = dir(fullfile(config.videoFolder, '*.mp4'));

if isempty(videoFiles)
    error('No video files found in %s', config.videoFolder);
end

fprintf('Found %d video(s) to process\n\n', length(videoFiles));

%% Step 2: Process each video
for vidIdx = 1:length(videoFiles)
    videoFile = videoFiles(vidIdx).name;
    videoPath = fullfile(config.videoFolder, videoFile);
    [~, videoName, ~] = fileparts(videoFile);
    
    fprintf('=== Processing Video %d/%d: %s ===\n', vidIdx, length(videoFiles), videoName);
    
    % Load video with resize and crop
    fprintf('  Loading video...\n');
    videoData = loadVideo(videoPath, config);
    
    % Create trajectory visualization
    fprintf('  Creating trajectory map...\n');
    trajectoryFig = createTrajectoryVisualization(videoData, config);
    
    % % Save figure
    % outputFilename = fullfile(config.outputFolder, sprintf('%s_trajectory.png', videoName));
    % saveas(trajectoryFig, outputFilename);
    % fprintf('  Saved: %s\n', outputFilename);
    
    % Save high-resolution version
    outputFilenameHD = fullfile(config.outputFolder, sprintf('%s_trajectory_HD.png', videoName));
    print(trajectoryFig, outputFilenameHD, '-dpng', '-r300');
    fprintf('  Saved HD: %s\n', outputFilenameHD);
    
    close(trajectoryFig);
    
    fprintf('  Complete!\n\n');
end

fprintf('=== All Videos Processed ===\n');
fprintf('Trajectory figures saved to: %s\n', config.outputFolder);

%% ========================================================================
%  FUNCTION DEFINITIONS
%% ========================================================================

function videoData = loadVideo(videoPath, config)
    % Load video and separate channels
    % Resizes to 512x512 if needed, then crops to 360x360 centered region
    
    vidObj = VideoReader(videoPath);
    
    % Get original dimensions
    originalHeight = vidObj.Height;
    originalWidth = vidObj.Width;
    numFrames = floor(vidObj.Duration * vidObj.FrameRate);
    
    fprintf('    Original dimensions: %dx%d\n', originalWidth, originalHeight);
    
    % Target size: resize to 512x512 if needed
    targetSize = 512;
    needsResize = (originalWidth ~= targetSize) || (originalHeight ~= targetSize);
    
    if needsResize
        fprintf('    Will resize to %dx%d\n', targetSize, targetSize);
    else
        fprintf('    Already %dx%d, no resize needed\n', targetSize, targetSize);
    end
    
    % Calculate 360x360 crop boundaries (centered at specified position on 512x512)
    cropSize = config.analysisCropSize; % 360
    centerX = config.cropCenterX;  % User-specified X center (default 256)
    centerY = config.cropCenterY;  % User-specified Y center (default 256)
    cropHalf = floor(cropSize / 2);   % 180
    
    cropXmin = centerX - cropHalf;  % e.g., 256 - 180 = 76
    cropXmax = centerX + cropHalf - 1;  % e.g., 256 + 180 - 1 = 435
    cropYmin = centerY - cropHalf;  % e.g., 256 - 180 = 76
    cropYmax = centerY + cropHalf - 1;  % e.g., 256 + 180 - 1 = 435
    
    actualCropHeight = cropYmax - cropYmin + 1;  % 360
    actualCropWidth = cropXmax - cropXmin + 1;   % 360
    
    fprintf('    Crop center: (%d, %d)\n', centerX, centerY);
    fprintf('    Cropping to [%d:%d, %d:%d] = %dx%d\n', ...
            cropYmin, cropYmax, cropXmin, cropXmax, ...
            actualCropWidth, actualCropHeight);
    
    % Pre-allocate arrays for cropped data
    greenChannel = zeros(actualCropHeight, actualCropWidth, numFrames, 'uint8');
    redChannel = zeros(actualCropHeight, actualCropWidth, numFrames, 'uint8');
    
    frameIdx = 1;
    while hasFrame(vidObj)
        frame = readFrame(vidObj);
        
        if needsResize
            % STEP 1: Resize to 512x512
            frameResized = imresize(frame, [targetSize, targetSize], 'bilinear');
        else
            % Already 512x512, no resize needed
            frameResized = frame;
        end
        
        % STEP 2: Crop to 360x360 centered region
        redChannel(:,:,frameIdx) = frameResized(cropYmin:cropYmax, cropXmin:cropXmax, 1);
        greenChannel(:,:,frameIdx) = frameResized(cropYmin:cropYmax, cropXmin:cropXmax, 2);
        
        frameIdx = frameIdx + 1;
        
        % Progress indicator
        if mod(frameIdx, 50) == 0
            fprintf('      Loaded %d frames...\n', frameIdx);
        end
    end
    
    % Trim to actual frames read
    greenChannel = greenChannel(:,:,1:frameIdx-1);
    redChannel = redChannel(:,:,1:frameIdx-1);
    
    videoData.green = greenChannel;
    videoData.red = redChannel;
    videoData.numFrames = frameIdx - 1;
    videoData.height = actualCropHeight;
    videoData.width = actualCropWidth;
    
    if needsResize
        fprintf('    Loaded %d frames (resized to 512x512, then cropped to %dx%d)\n', ...
                videoData.numFrames, actualCropWidth, actualCropHeight);
    else
        fprintf('    Loaded %d frames (cropped to %dx%d, no resize)\n', ...
                videoData.numFrames, actualCropWidth, actualCropHeight);
    end
end

function fig = createTrajectoryVisualization(videoData, config)
    % Create cumulative trajectory visualization using alpha blending
    %
    % Green channel → Gray (0.5, 0.5, 0.5) with alpha up to 0.8
    % Red channel → Blue (0, 0, 1) with alpha up to 0.8
    %
    % Alpha blending formula: result = background * (1-α) + color * α
    % Maximum alpha = 0.8 when signal present in ALL frames
    
    height = videoData.height;
    width = videoData.width;
    numFrames = videoData.numFrames;
    
    % Initialize accumulation maps (count how many times each pixel is active)
    greenAccumulation = zeros(height, width);
    redAccumulation = zeros(height, width);
    
    fprintf('    Processing %d frames...\n', numFrames);
    
    % Accumulate signal presence across all frames
    for frameIdx = 1:numFrames
        % Get current frame
        greenFrame = double(videoData.green(:,:,frameIdx));
        redFrame = double(videoData.red(:,:,frameIdx));
        
        % Find pixels above threshold (signal present)
        greenActive = greenFrame > config.greenThreshold;
        
        % For red: Check both threshold AND purity (red > green * ratio)
        % This ensures we only count PURE red, not white or pink
        redAboveThreshold = redFrame > config.redThreshold;
        redDominant = redFrame > (greenFrame * config.redPurityRatio);
        redActive = redAboveThreshold & redDominant;
        
        % Accumulate counts
        greenAccumulation = greenAccumulation + double(greenActive);
        redAccumulation = redAccumulation + double(redActive);
        
        % Progress indicator
        if mod(frameIdx, 50) == 0
            fprintf('      Processed %d/%d frames\n', frameIdx, numFrames);
        end
    end
    
    fprintf('    Accumulation complete\n');
    
    % Normalize accumulation maps to [0, 1] range
    % More appearances → Higher value → More opaque
    greenNormalized = greenAccumulation / numFrames;  % 0-1 range (proportion of frames)
    redNormalized = redAccumulation / numFrames;
    
    % Scale to maximum alpha of 0.8
    % If a pixel appears in ALL frames, alpha = 0.8
    maxAlpha = 0.8;
    greenAlpha = greenNormalized * 0.99;
    redAlpha = redNormalized * 0.9;
    
    % Create RGB image for visualization
    % Start with white background (1, 1, 1)
    rgbImage = ones(height, width, 3);
    
    % Define colors
    grayColor = [0.55, 0.55, 0.55];   % Medium gray for vessel
    blueColor = [0, 0.2, 1];          % Pure blue for platelet
    
    % Apply gray for green channel trajectory using alpha blending
    % Formula: result = background * (1 - alpha) + color * alpha
    for i = 1:height
        for j = 1:width
            if greenAlpha(i,j) > 0
                alpha = greenAlpha(i,j);
                % Alpha blend gray onto white background
                rgbImage(i,j,1) = rgbImage(i,j,1) * (1 - alpha) + grayColor(1) * alpha;
                rgbImage(i,j,2) = rgbImage(i,j,2) * (1 - alpha) + grayColor(2) * alpha;
                rgbImage(i,j,3) = rgbImage(i,j,3) * (1 - alpha) + grayColor(3) * alpha;
            end
        end
    end
    
    % Apply blue for red channel trajectory using alpha blending
    % This blends onto whatever is already there (white or gray)
    for i = 1:height
        for j = 1:width
            if redAlpha(i,j) > 0
                alpha = redAlpha(i,j);
                % Alpha blend blue onto current color
                rgbImage(i,j,1) = rgbImage(i,j,1) * (1 - alpha) + blueColor(1) * alpha;
                rgbImage(i,j,2) = rgbImage(i,j,2) * (1 - alpha) + blueColor(2) * alpha;
                rgbImage(i,j,3) = rgbImage(i,j,3) * (1 - alpha) + blueColor(3) * alpha;
            end
        end
    end
    
    fprintf('    Creating visualization...\n');
    
    % Create figure
    fig = figure('Position', [100, 100, 370, 370]);
    
    % Main trajectory plot
    imshow(rgbImage);
    
    fprintf('    Visualization complete\n');
end
