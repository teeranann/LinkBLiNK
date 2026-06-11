% create_overlay_video_matlab.m
% This MATLAB function loads TIFF frames and trajectory data,
% then overlays trajectories onto the video and saves it.
%
% This version has drawing functions commented out for troubleshooting.
%
% This function is designed to be called from Python via command-line.
%
% Usage:
% create_overlay_video_matlab(image_folder_path, trajectory_csv_path, output_video_path, ...
%                             fps, particle_color_str, trajectory_color_str, ...
%                             line_thickness, dot_radius, display_id_flag)
%
% Arguments:
%   image_folder_path:      Full path to the folder containing TIFF image frames.
%   trajectory_csv_path:    Full path to the CSV file containing linked trajectory data (e.g., from Trackpy).
%   output_video_path:      Full path for the output AVI video file.
%   fps:                    Frames per second for the output video.
%   particle_color_str:     String representation of RGB color for particle dots (e.g., '[1,0,0]' for red).
%   trajectory_color_str:   String representation of RGB color for trajectory lines (e.g., '[0,0,1]' for blue).
%   line_thickness:         Thickness of the trajectory lines.
%   dot_radius:             Radius of the particle dots.
%   display_id_flag:        Flag (0 or 1) to determine if particle IDs should be displayed.

function create_overlay_video_matlab(image_folder_path, trajectory_csv_path, output_video_path, ...
                                    fps, particle_color_str, trajectory_color_str, ...
                                    line_thickness, dot_radius, display_id_flag)

    disp('MATLAB: Starting video overlay generation...');
    disp(['Image Folder: ', image_folder_path]);
    disp(['Trajectory CSV: ', trajectory_csv_path]);
    disp(['Output Video: ', output_video_path]);

    % --- 1. Validate Inputs ---
    if ~isfolder(image_folder_path)
        error('MATLAB:InvalidInput', 'Image folder does not exist: %s', image_folder_path);
    end
    if ~isfile(trajectory_csv_path)
        error('MATLAB:InvalidInput', 'Trajectory CSV file does not exist: %s', trajectory_csv_path);
    end

    % Convert color strings to numeric arrays (expected [0-1] for im2double image)
    particle_color_rgb = str2num(particle_color_str);
    trajectory_color_rgb = str2num(trajectory_color_str);

    % --- 2. Load Data ---
    fileList = dir(fullfile(image_folder_path, '*.tif')); % Assuming TIFFs, adjust if needed
    if isempty(fileList)
        fileList = dir(fullfile(image_folder_path, '*.png')); % Try PNG as fallback
    end
    if isempty(fileList)
        error('MATLAB:NoImagesFound', 'No TIFF or PNG image files found in %s', image_folder_path);
    end
    
    % Get image dimensions from the first image
    if isempty(fileList)
        error('MATLAB:NoImagesFound', 'Cannot determine video dimensions, no image files in %s', image_folder_path);
    end
    tempImg = imread(fullfile(image_folder_path, fileList(1).name));
    [img_height, img_width, ~] = size(tempImg);

    % Load trajectory data (still needed for overall frame count)
    try
        T = readtable(trajectory_csv_path);
        if ~ismember({'frame', 'particle', 'x', 'y'}, T.Properties.VariableNames)
            error('MATLAB:CSVFormatError', 'Required columns (frame, particle, x, y) not found in CSV.');
        end
        trajectory_data = T{:, {'frame', 'particle', 'x', 'y'}};
    catch ME
        error('MATLAB:CSVReadError', 'Error reading trajectory CSV: %s', ME.message);
    end

    allFrames = unique(trajectory_data(:,1));
    numLinkedParticles = length(unique(trajectory_data(:,2)));
    disp(['MATLAB: Loaded ', num2str(numLinkedParticles), ' linked particles across ', num2str(length(allFrames)), ' frames.']);

    % --- 3. Setup Video Writer ---
    try
        % FIX: Specify Width and Height directly in the VideoWriter constructor
        outputVideoObj = VideoWriter(output_video_path, 'Motion JPEG AVI');
        outputVideoObj.FrameRate = fps;
        % outputVideoObj.Width = img_width;  % REMOVED: These lines are the problem
        % outputVideoObj.Height = img_height; % REMOVED: These lines are the problem
        open(outputVideoObj); % Now open() will initialize with the correct properties
    catch ME
        error('MATLAB:VideoWriterError', 'Could not initialize VideoWriter: %s. Check codecs or file path.', ME.message);
    end

    % --- 4. Process Frames (Drawing functions COMMENTED OUT) ---
    progressBar = waitbar(0, 'Creating Overlay Video...');
    
    for k = 1:length(fileList)
        currentFrameFilename = fileList(k).name;
        currentFramePath = fullfile(image_folder_path, currentFrameFilename);
        
        [~, name, ext] = fileparts(currentFrameFilename);
        frameNumStr = regexp(name, '\d+$', 'match', 'once');
        if isempty(frameNumStr)
             frameNumStr = name;
        end
        current_frame_number = str2double(frameNumStr);
        
        if isnan(current_frame_number)
            warning('MATLAB:FrameNumParse', 'Could not parse frame number from %s. Skipping frame.', currentFrameFilename);
            continue;
        end

        img = imread(currentFramePath);
        
        % FIX: Convert any image type to uint8 for VideoWriter compatibility
        if isinteger(img)
            img_to_write = im2uint8(img); % Convert to uint8, scaling if necessary
        elseif isfloat(img)
            img_to_write = im2uint8(img); % Convert to uint8, assuming 0-1 or 0-255 float range
        else
            warning('MATLAB:UnsupportedType', 'Unsupported image type for video writing. Attempting conversion to uint8.');
            img_to_write = im2uint8(img); % Fallback
        end

        if size(img_to_write,3) == 1 % Convert grayscale to RGB if needed for color video
            img_to_write = repmat(img_to_write, [1 1 3]);
        end

        % Drawing lines (COMMENTED OUT FOR TESTING)
        % Drawing current particle positions and IDs (COMMENTED OUT FOR TESTING)

        writeVideo(outputVideoObj, img_to_write); % Write the frame to video
        waitbar(k / length(fileList), progressBar, sprintf('Creating Overlay Video: Frame %d/%d', k, length(fileList)));
    end

    close(progressBar);
    close(outputVideoObj);
    disp('MATLAB: Video overlay generation complete!');
end