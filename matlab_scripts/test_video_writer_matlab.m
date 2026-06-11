% test_video_writer_matlab.m
% A simple MATLAB script to test VideoWriter functionality in batch mode.

function test_video_writer_matlab(output_file_path)
    disp(['MATLAB Test: Attempting to write to: ', output_file_path]);
    try
        % Specify size explicitly in constructor, and ensure RGB for common codecs
        writerObj = VideoWriter(output_file_path, 'Motion JPEG AVI');
        writerObj.FrameRate = 10;
        
        % Set dimensions for the test video (e.g., 100x100)
        % Note: VideoWriter properties like Width/Height are set via the constructor arguments if the profile supports it,
        % or inferred from the first frame. Explicitly passing size to constructor is not typically done for 'Motion JPEG AVI'
        % in the same way as, say, 'MPEG-4', but it needs to be inferred from the frames.
        % The previous error was about *setting* it as a property *after* creation.
        % We will just ensure frames are RGB uint8 and rely on writeVideo to infer.
        
        open(writerObj);
        
        % Write 5 dummy frames (100x100 RGB uint8)
        for i = 1:5
            img = uint8(rand(100, 100, 3) * 255); % Random 100x100 RGB uint8 image
            writeVideo(writerObj, img);
            disp(['MATLAB Test: Wrote frame ', num2str(i)]);
        end
        
        close(writerObj);
        disp('MATLAB Test: Video writer closed successfully.');
    catch ME
        disp(['MATLAB Test ERROR: ', ME.message]);
        % IMPORTANT: Rethrow the error so Python can catch the specific MATLAB error message
        rethrow(ME);
    end
end