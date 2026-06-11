% seq_to_tif.m
% This MATLAB function converts a single .seq file to individual TIFF frames.
% It is designed to be called from a Python script via command-line.

function seq_to_tif(input_seq_filepath, output_tif_directory)
    % Ensure input_seq_filepath and output_tif_directory are strings
    if ~ischar(input_seq_filepath) || ~ischar(output_tif_directory)
        error('seq_to_tif:InvalidInput', 'Inputs must be strings.');
    end

    disp(['MATLAB: Starting conversion of ', input_seq_filepath]);

    % Get the base name of the sequence file (without extension) for frame naming
    [~, seqBaseName, ~] = fileparts(input_seq_filepath);

    % Ensure the output directory exists
    if ~isfolder(output_tif_directory)
        mkdir(output_tif_directory);
        disp(['MATLAB: Created output folder: ', output_tif_directory]);
    end

    try
        % Read the .seq file
        % IMPORTANT: You need to have the 'readSEQ' function available in your MATLAB path.
        % This typically comes from a library like Bio-Formats or a custom script.
        % If 'readSEQ' is not a built-in function or on your path, this will fail.
        data = readSEQ(input_seq_filepath);

        % Check if 'data.image' is empty
        if isempty(data.image)
            warning('seq_to_tif:EmptyImage', 'No image data found in %s. Skipping.', input_seq_filepath);
            return; % Exit function if no image data
        end

        % Get the dimensions of the loaded image data
        % data.image is typically H x W x (C) x F (for grayscale, C is absorbed or 1)
        
        % Ensure imgSequence is 4D (Height x Width x Channel x Frames) for consistent indexing
        if ndims(data.image) == 3 % If it's H x W x Frames (grayscale)
            [H, W, F] = size(data.image);
            imgSequence = reshape(data.image, H, W, 1, F); % Add singleton dimension for color channel
        else % Already H x W x C x F
            imgSequence = data.image;
        end

        numFrames = size(imgSequence, 4);
        disp(['MATLAB: Converting ', num2str(numFrames), ' frames from ', seqBaseName, '...']);

        % Loop through each frame and save it as an individual TIFF file
        for frameIdx = 1:numFrames
            % Get the current frame
            currentFrame = imgSequence(:, :, :, frameIdx);
            
            % If it's a color image (3 channels), ensure it's converted to grayscale if U-Net expects 1 channel
            if size(currentFrame, 3) > 1
                currentFrame = rgb2gray(currentFrame);
            end
            
            % Construct the output filename for the individual frame
            % Naming: original_seq_name_frame_0001.tif
            frameOutputFileName = sprintf('%s_frame_%04d.tif', seqBaseName, frameIdx);
            fullFrameOutputPath = fullfile(output_tif_directory, frameOutputFileName);

            % Save the frame as a TIFF
            % 'Compression', 'none' is generally good for scientific data
            % Adjust 'BitDepth' if your data is consistently 16-bit
            if isinteger(currentFrame) % Check if integer (e.g., uint8, uint16)
                imwrite(currentFrame, fullFrameOutputPath, 'Compression', 'none');
            else % Assume float, convert to uint16 or uint8 if necessary for saving
                % Normalize float data to appropriate range before saving as integer TIFF
                currentFrame_norm = currentFrame - min(currentFrame(:));
                if max(currentFrame_norm(:)) > 0
                    currentFrame_norm = currentFrame_norm / max(currentFrame_norm(:));
                end

                if isa(data.image, 'uint16') || max(currentFrame_norm(:)) > 1 % Original was 16-bit or large range
                    imwrite(uint16(currentFrame_norm * 65535), fullFrameOutputPath, 'Compression', 'none');
                else % Assume 8-bit equivalent
                    imwrite(uint8(currentFrame_norm * 255), fullFrameOutputPath, 'Compression', 'none');
                end
            end
        end

        disp(['MATLAB: Successfully converted all frames from: ', seqBaseName, ' to individual TIFFs in ', output_tif_directory]);
    catch ME
        % Catch any errors during processing of a single file,
        warning('seq_to_tif:ConversionError', 'Error converting %s: %s', input_seq_filepath, ME.message);
    end

    % Optional: Clean up temporary JPEG files created by readSEQ (if any, typically for older versions)
    % This part assumes temporary files are created in the current working directory of MATLAB.
    tempFiles = {'_tmp1.jpg', '_tmp2.jpg', '_tmp3.jpg'};
    for j = 1:length(tempFiles)
        if exist(tempFiles{j}, 'file')
            delete(tempFiles{j});
        end
    end
    disp('MATLAB: Batch conversion function finished.');
end