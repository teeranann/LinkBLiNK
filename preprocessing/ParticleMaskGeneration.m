% MATLAB Script: Interactive Particle Labeling and Mask Generation
% This script allows users to manually click on particles in images,
% automatically generates precise masks using 2D Gaussian fitting,
% and organizes images and masks into categorized folders.
%
% Requirements:
%   - MATLAB (tested with R2014b and later)
%   - Optimization Toolbox (for lsqcurvefit used in cntr2dg)
%   - Your raw image files (.tif, .png, .jpg, etc.)
%
% Instructions:
% 1. Run the script.
% 2. Select the folder containing your raw SMLM images.
% 3. Optionally, select an output base folder (or it will default to a new folder).
% 4. For each image displayed:
%    - Left-click on the approximate center of each particle you want to label.
%    - A green circle will mark the detected sub-pixel center.
%    - A white rectangle will show the automatically generated Gaussian fitting ROI.
%    - The generated binary mask will be overlaid in semi-transparent blue.
%    - Press the RIGHT-CLICK mouse button when you are done labeling ALL
%      particles in the current image.
% 5. The script will automatically save the original labeled image and its
%    combined binary mask (0/255) into separate folders.
% 6. Images for which no particles were labeled will be moved to an
%    'Unlabeled_Images' folder.

close all; % Close all open figures
clear;     % Clear workspace variables

%% --- Configuration Parameters (Adjust as needed) ---

% Default parameters for particle detection and mask generation
% pxl: Half-width of the square ROI (2*pxl+1 x 2*pxl+1) for initial max intensity search around click.
%      Should be large enough to catch the particle, but not so large it overlaps too much.
pxl = 7; % Reduced from 10, common range is 5-7 for typical PSFs.

% gpxl: Half-width of the square ROI (2*gpxl+1 x 2*gpxl+1) for 2D Gaussian fitting.
%       Should encompass most of the particle's signal without too much background/other particles.
%       Typically 2-3 times your expected particle sigma (spread).
gpxl = 7; % Reduced from 10, common range 5-7.

% bp1: Bandpass filter lower cutoff frequency (pixels). Filters out large-scale background variations.
%      Increase if your background is very uneven. Too high may affect particle signal.
bp1 = 1.5; % Slightly increased from 1, common is 1-3.

% bp2: Bandpass filter upper cutoff frequency (pixels). Filters out high-frequency noise.
%      Decrease if your images are very noisy. Too low may blur particles.
bp2 = 8;   % Reduced from 30, common range is 3-15 for typical SMLM particles.

% Mask generation parameters
% mask_radius_factor: This factor determines the radius of the mask around the fitted Gaussian center.
% A value of 2.5 means the mask will cover approximately 2.5 standard deviations (sigma)
% of the Gaussian, capturing most of its intensity. Adjust based on desired mask size.
mask_radius_factor = 2.5;

% mask_threshold_factor: This threshold is applied to the image region within the automatically
% defined particle mask. Pixels above this percentage of the local peak
% intensity will be included in the binary mask (0-255).
% Lower value -> wider mask, Higher value -> tighter mask.
mask_threshold_factor = 0.3; % 0.3 = 30% of local peak intensity

% Parameters for the Gaussian fitting function (cntr2dg)
% initial_Wg_guess: Initial guess for FWHM (Full Width at Half Maximum) in pixels.
%                   Should be a reasonable estimate of your particle size (e.g., 2-5 pixels).
initial_Wg_guess = 3; % Common FWHM range for SMLM particles.

% initial_Ag_guess_factor: Factor to multiply max intensity of ROI for initial Amplitude guess.
initial_Ag_guess_factor = 0.8; % A bit aggressive, assuming click is on a bright particle.

% File extensions to look for
image_extensions = {'*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg'};

% Debugging plot for Gaussian fit (set to true to show a separate plot for each fit)
ENABLE_GAUSSIAN_DEBUG_PLOT = false;

%% --- Folder Selection ---

% 1. Select Input Folder
input_folder = uigetdir(pwd, 'Select Folder Containing Raw SMLM Images');
if input_folder == 0
    disp('No input folder selected. Script aborted.');
    return;
end
fprintf('Input Images Folder: %s\n', input_folder);

% 2. Select Output Base Folder
output_base_folder = uigetdir(pwd, 'Select Output Folder for Labeled Data (or create new)');
if output_base_folder == 0
    % If user cancels, create a default output folder in current directory
    output_base_folder = fullfile(pwd, 'Particle_Labeling_Output');
    fprintf('No output folder selected. Creating default: %s\n', output_base_folder);
end

% Create the base output folder if it doesn't exist
if ~exist(output_base_folder, 'dir')
    mkdir(output_base_folder);
end

% Define output subfolders
labeled_images_folder = fullfile(output_base_folder, 'Labeled_Images');
generated_masks_folder = fullfile(output_base_folder, 'Generated_Masks');
unlabeled_images_folder = fullfile(output_base_folder, 'Unlabeled_Images');

% Create output subfolders
mkdir(labeled_images_folder);
mkdir(generated_masks_folder);
mkdir(unlabeled_images_folder);

fprintf('Output Labeled Images Folder: %s\n', labeled_images_folder);
fprintf('Output Generated Masks Folder: %s\n', generated_masks_folder);
fprintf('Output Unlabeled Images Folder: %s\n', unlabeled_images_folder);

%% --- Image Loading and Iteration ---

% Get list of all image files in the input folder
image_files = [];
for ext_idx = 1:length(image_extensions)
    current_files = dir(fullfile(input_folder, image_extensions{ext_idx}));
    image_files = [image_files; current_files]; %#ok<AGROW>
end

if isempty(image_files)
    disp('No image files found in the selected input folder. Script aborted.');
    return;
end

fprintf('Found %d image files to process.\n', length(image_files));

% Setup the figure for interactive labeling
hFig = figure('units','normalized','position',[0.1 0.1 0.7 0.8], 'Name', 'Particle Labeler');
hAx = axes('Parent', hFig, 'Position', [0.05 0.05 0.9 0.9]); % Make axes fill most of the figure

for i = 1:length(image_files)
    current_filename = image_files(i).name;
    full_image_path = fullfile(input_folder, current_filename);

    % Read the image
    img_original = imread(full_image_path);
    if isempty(img_original)
        fprintf('Warning: Could not read image %s. Skipping.\n', current_filename);
        continue;
    end
    
    % Ensure image is grayscale for processing
    if size(img_original, 3) == 3
        img_original = rgb2gray(img_original);
    end

    % Convert to double for processing (important for calculations)
    img_original_double = double(img_original);
    [dim_y, dim_x] = size(img_original_double);

    % Simple global background estimation (median intensity)
    current_background = median(img_original_double(:));

    % Display the current image
    imagesc(hAx, img_original_double);
    colormap(hAx, hot); % Use 'hot' colormap
    axis(hAx, 'xy');
    set(hAx, 'dataaspectratio', [1 1 1]); % Square pixels
    title(hAx, sprintf('Labeling: %s (Left-click particles, Right-click to finish)', current_filename), 'Interpreter', 'none');
    xlabel(hAx, 'X (pixels)');
    ylabel(hAx, 'Y (pixels)');
    colorbar(hAx); % Add colorbar for intensity scale
    drawnow;
    
    hold(hAx, 'on'); % Allow drawing on top of the image
    
    % Initialize an empty mask for the current image (will accumulate all labeled particles)
    current_image_binary_mask = false(dim_y, dim_x); % Logical mask
    particle_counter = 0; % Counter for particles labeled in the current image

    fprintf('Processing %s: Left-click on particle centers. Right-click to finish.\n', current_filename);

    button = 1;
    while button ~= 3 % Loop until right-click (button 3)
        % Check if figure was closed by user
        if ~ishandle(hFig)
            disp('Figure window closed. Exiting labeling process.');
            return; % Exit the entire script
        end

        % Ensure figure and axes are current for ginput
        figure(hFig); % Make hFig the current figure
        axes(hAx);    % Make hAx the current axes
        
        try
            [x_click, y_click, button] = ginput(1); % Get one click
        catch ME
            fprintf('Error during ginput: %s. Likely user closed figure. Exiting loop for current image.\n', ME.message);
            button = 3; % Simulate right-click to break loop gracefully
            continue; % Go to next iteration to check button == 3 condition
        end

        if button == 3 % Right-click: done with this image
            break;
        end

        % Ensure click is within image bounds
        x_click_round = round(x_click);
        y_click_round = round(y_click);

        if x_click_round < 1 || x_click_round > dim_x || y_click_round < 1 || y_click_round > dim_y
            fprintf('  Click outside image bounds. Please click within the image. Skipping this click.\n');
            continue;
        end

        % --- Particle Detection Logic ---
        % Define ROI for initial max intensity search
        x_min_search_roi = max(1, x_click_round - pxl);
        x_max_search_roi = min(dim_x, x_click_round + pxl);
        y_min_search_roi = max(1, y_click_round - pxl);
        y_max_search_roi = min(dim_y, y_click_round + pxl);

        % Apply bandpass filter to the image
        H_filter = mkffilt(img_original_double, bp1, bp2);
        img_filtered = fpass(img_original_double, H_filter);

        % Find local max within the search ROI in the filtered image
        roi_filtered_search = img_filtered(y_min_search_roi:y_max_search_roi, x_min_search_roi:x_max_search_roi);
        
        if isempty(roi_filtered_search) || all(isnan(roi_filtered_search(:))) || all(roi_filtered_search(:) == 0)
            fprintf('  Warning: Search ROI for click is empty or invalid. Skipping this click.\n');
            continue;
        end
        
        [~, max_col_idx] = max(max(roi_filtered_search, [], 1));
        [~, max_row_idx] = max(max(roi_filtered_search, [], 2));
        
        % Convert local ROI coordinates back to global image coordinates
        x_max_global = x_min_search_roi - 1 + max_col_idx;
        y_max_global = y_min_search_roi - 1 + max_row_idx;

        % Initial amplitude guess from the max of the filtered ROI
        initial_Ag_for_fit = max(roi_filtered_search(:)) * initial_Ag_guess_factor;
        initial_Ag_for_fit = max(initial_Ag_for_fit, 10); % Ensure it's not too low

        % Perform 2D Gaussian fitting
        try
            % cntr2dg fits to intensity above background.
            % Pass the full image to cntr2dg, it will extract its own sub-ROI.
            ctt = cntr2dg(img_original_double - current_background, [x_max_global, y_max_global], gpxl, initial_Wg_guess, initial_Ag_for_fit, ENABLE_GAUSSIAN_DEBUG_PLOT);
            particle_x = ctt.xywi(1,1);
            particle_y = ctt.xywi(1,2);
            fitted_Wg = ctt.xywi(1,3); % Fitted FWHM
            % fitted_Ag = ctt.xywi(1,4); % Fitted Amplitude

            % Basic validation of fit results (e.g., sensible position and width)
            if isnan(particle_x) || isnan(particle_y) || fitted_Wg < 0.5 || fitted_Wg > 15
                 error('Invalid fit parameters detected. Particle too small, too large, or fit diverged.');
            end

        catch ME
            fprintf('  Error during Gaussian fitting for click at (%.1f, %.1f): %s. Skipping this click.\n', x_click, y_click, ME.message);
            fprintf('  Check if Optimization Toolbox is installed and parameters are suitable for fitting.\n');
            continue; % Skip to next click if fitting fails
        end

        % --- Create Mask for this particle based on Gaussian fit ---
        % Estimate sigma from FWHM (Wg) for mask radius
        sigma_val = fitted_Wg / (2 * sqrt(2 * log(2))); % sigma = FWHM / 2.355
        mask_radius_px = ceil(sigma_val * mask_radius_factor); % Mask covers 'mask_radius_factor' sigmas

        % Define a square bounding box for the mask around the sub-pixel center
        mask_x_min = max(1, round(particle_x) - mask_radius_px);
        mask_x_max = min(dim_x, round(particle_x) + mask_radius_px);
        mask_y_min = max(1, round(particle_y) - mask_radius_px);
        mask_y_max = min(dim_y, round(particle_y) + mask_radius_px);
        
        % Ensure mask region is valid
        if mask_x_min > mask_x_max || mask_y_min > mask_y_max || mask_x_min > dim_x || mask_y_min > dim_y
            fprintf('  Warning: Calculated mask region for particle at (%.1f, %.1f) is invalid or outside image bounds. Skipping mask for this click.\n', particle_x, particle_y);
            continue;
        end

        % Extract ROI from original image (not filtered) for thresholding
        roi_for_mask_thresh = img_original_double(mask_y_min:mask_y_max, mask_x_min:mask_x_max);

        % Apply thresholding within this ROI
        if ~isempty(roi_for_mask_thresh) && max(roi_for_mask_thresh(:)) > 0
            % Threshold based on a percentage of the peak intensity in the ROI
            threshold_value_for_mask = max(roi_for_mask_thresh(:)) * mask_threshold_factor;
            binary_roi_mask = (roi_for_mask_thresh > threshold_value_for_mask);
            
            % Add this particle's mask to the overall image mask (logical OR)
            current_image_binary_mask(mask_y_min:mask_y_max, mask_x_min:mask_x_max) = ...
                current_image_binary_mask(mask_y_min:mask_y_max, mask_x_min:mask_x_max) | binary_roi_mask;
            
            particle_counter = particle_counter + 1;

            % --- Visualize the detected particle and its mask box ---
            plot(hAx, particle_x, particle_y, 'go', 'MarkerSize', 8, 'LineWidth', 1.5); % Green circle for sub-pixel center
            rectangle(hAx, 'Position', [mask_x_min, mask_y_min, (mask_x_max - mask_x_min), (mask_y_max - mask_y_min)], 'EdgeColor', 'w', 'LineWidth', 1); % White box
            text(hAx, particle_x + mask_radius_px + 2, particle_y, num2str(particle_counter), 'Color', [0 1 0], 'FontSize', 10); % Green ID
            
            % Overlay generated mask with transparency
            hMaskOverlay = imagesc(hAx, current_image_binary_mask);
            set(hMaskOverlay, 'AlphaData', current_image_binary_mask * 0.3); % 30% transparency where mask is true
            
            drawnow; % Update display
        else
            fprintf('  Warning: ROI for mask creation is empty or all zero. Skipping mask for this click.\n');
        end
    end
    
    hold(hAx, 'off'); % Release hold on axes for the next image
    
    % --- Save the accumulated mask and original image ---
    if particle_counter > 0
        % Convert logical mask to uint8 (0 or 255) for saving as image
        final_mask_uint8 = uint8(current_image_binary_mask * 255);
        
        % Create unique filenames for mask and original image copy
        [~, name, ext] = fileparts(current_filename);
        mask_output_filename = [name '_mask.png']; % Always save masks as PNG
        original_copy_filename = [name ext]; % Keep original extension for image copy
        
        imwrite(final_mask_uint8, fullfile(generated_masks_folder, mask_output_filename));
        imwrite(img_original, fullfile(labeled_images_folder, original_copy_filename));
        fprintf('Saved %d particles for "%s". Mask to "%s", image to "%s".\n', ...
            particle_counter, current_filename, mask_output_filename, original_copy_filename);
    else
        % If no particles were labeled, move the original image to 'Unlabeled_Images'
        fprintf('No particles labeled in "%s". Moving to Unlabeled_Images folder.\n', current_filename);
        % Use fullfile for source and destination paths to ensure correctness
        movefile(full_image_path, fullfile(unlabeled_images_folder, current_filename));
    end
    
    cla(hAx); % Clear axes for the next image
    
    % Check if the user closed the figure window
    if ~ishandle(hFig)
        fprintf('Figure window closed. Exiting labeling process.\n');
        break; % Exit the main loop
    end
end

fprintf('\nInteractive labeling process completed for all images.\n');

%% --- Helper Functions (Nested Functions for Independence) ---
% These functions are included here to make the script self-contained.
% They are simplified versions and may need fine-tuning for your specific data.

function H = mkffilt(img, bp1, bp2)
    % mkffilt: Creates a 2D Fourier bandpass filter.
    % img: Input image (used for size)
    % bp1: Lower cutoff frequency (pixels)
    % bp2: Upper cutoff frequency (pixels)

    [M, N] = size(img);
    cx = floor(N/2) + 1;
    cy = floor(M/2) + 1;

    [X, Y] = meshgrid(1:N, 1:M);
    R = sqrt((X - cx).^2 + (Y - cy).^2); % Radial distance from center

    % Create bandpass filter (Gaussian-like rolloff for smoothness)
    H_low = 1 ./ (1 + (R ./ bp1).^2); % Low-pass component
    H_high = 1 ./ (1 + (bp2 ./ R).^2); % High-pass component
    H = H_low .* H_high;
    H(isnan(H)) = 0; % Handle division by zero at R=0
end

function filtered_img = fpass(img, H)
    % fpass: Applies a Fourier filter (H) to an image (img).
    % img: Input image
    % H: Fourier filter created by mkffilt

    F = fftshift(fft2(img)); % Fourier transform and shift zero frequency to center
    filtered_F = F .* H;     % Apply the filter
    filtered_img = real(ifft2(ifftshift(filtered_F))); % Inverse transform and take real part
end

function ctt = cntr2dg(img_double, initial_xy, gpxl, initial_Wg, initial_Ag, enable_debug_plot)
    % cntr2dg: Fits a 2D Gaussian to an image ROI for sub-pixel localization.
    % img_double: The original image (double precision), with global background potentially subtracted.
    % initial_xy: [x, y] initial guess for particle center (from max pixel)
    % gpxl: Radius for fitting window (ROI size is 2*gpxl+1)
    % initial_Wg: Initial guess for FWHM (Full Width at Half Maximum)
    % initial_Ag: Initial guess for Amplitude (peak intensity)
    % enable_debug_plot: Boolean to control a separate debug plot for each fit.
    %
    % Output ctt: struct containing xywi, which is [x, y, Wg, Ag, R2, RMSE]

    % Define the fitting window around the initial guess
    x_center_initial = initial_xy(1);
    y_center_initial = initial_xy(2);

    [dim_y, dim_x] = size(img_double);

    x_min_fit = max(1, round(x_center_initial) - gpxl);
    x_max_fit = min(dim_x, round(x_center_initial) + gpxl);
    y_min_fit = max(1, round(y_center_initial) - gpxl);
    y_max_fit = min(dim_y, round(y_center_initial) + gpxl);

    % Extract the actual data for fitting from the full image
    fit_data = img_double(y_min_fit:y_max_fit, x_min_fit:x_max_fit);

    % Ensure fitting data is not empty or invalid
    if isempty(fit_data) || all(isnan(fit_data(:)))
        error('Fitting ROI is empty or contains only NaNs. Check gpxl and click location.');
    end

    % Create local meshgrid for fitting
    [X_local, Y_local] = meshgrid(x_min_fit:x_max_fit, y_min_fit:y_max_fit);

    % Convert FWHM to sigma for Gaussian equation: sigma = FWHM / (2 * sqrt(2 * log(2)))
    initial_sigma = initial_Wg / (2 * sqrt(2 * log(2)));

    % Initial parameters for the 2D Gaussian fit: [Amplitude, x0, y0, sigma_x, sigma_y, Offset]
    p0 = [initial_Ag, x_center_initial, y_center_initial, initial_sigma, initial_sigma, min(fit_data(:))]; % Use local min for initial offset guess

    % Define the 2D Gaussian function
    gaussian2D = @(p, xdata) p(1) * exp(-((xdata(:,1) - p(2)).^2 / (2 * p(4)^2) + ...
                                         (xdata(:,2) - p(3)).^2 / (2 * p(5)^2))) + p(6);

    % Prepare data for lsqcurvefit
    xdata_fit = [X_local(:), Y_local(:)];
    ydata_fit = fit_data(:);

    % Set lower and upper bounds for parameters (important for robust fitting)
    % Amplitude: must be positive, reasonable max (e.g., 2^16-1 for 16-bit)
    % x0, y0: within fitting ROI
    % sigma_x, sigma_y: minimum 0.5 (for very sharp features), maximum 10 (or higher if particles are very blurry)
    % Offset: can be bounded from 0 up to max intensity of surrounding dark area.
    lb = [1, x_min_fit, y_min_fit, 0.5, 0.5, -100]; % min Amplitude 1, min sigma 0.5
    ub = [inf, x_max_fit, y_max_fit, 10, 10, inf]; % max sigma 10 (adjust ub(4),ub(5) based on max expected particle size)

    % Use lsqcurvefit for fitting (Requires Optimization Toolbox)
    options = optimoptions('lsqcurvefit', 'Display', 'off');
    [p_fit, resnorm, ~, exitflag, output] = lsqcurvefit(gaussian2D, p0, xdata_fit, ydata_fit, lb, ub, options);

    if exitflag <= 0 % If fitting failed or did not converge
        % Fallback to initial guess or simple max if fit fails
        warning('Gaussian fitting failed or did not converge. Using initial guess for position and width for a click at (%.1f, %.1f).', initial_xy(1), initial_xy(2));
        fitted_x = initial_xy(1);
        fitted_y = initial_xy(2);
        fitted_Wg = initial_Wg; % Revert to initial Wg guess
        fitted_Ag = initial_Ag; % Use initial Ag for output
        R_squared = 0;
        RMSE = inf;
    else
        fitted_x = p_fit(2);
        fitted_y = p_fit(3);
        fitted_sigma_x = p_fit(4);
        % fitted_sigma_y = p_fit(5); % If fitting anisotropic, use this.
        fitted_Ag = p_fit(1);
        fitted_offset = p_fit(6);

        % Convert fitted sigma back to FWHM: FWHM = sigma * (2 * sqrt(2 * log(2)))
        fitted_Wg = fitted_sigma_x * (2 * sqrt(2 * log(2))); % Assuming isotropic for output Wg

        % Calculate R-squared and RMSE for fit quality
        y_predicted = gaussian2D(p_fit, xdata_fit);
        SS_total = sum((ydata_fit - mean(ydata_fit)).^2);
        SS_residual = sum((ydata_fit - y_predicted).^2);
        R_squared = 1 - (SS_residual / SS_total);
        RMSE = sqrt(SS_residual / numel(ydata_fit));
    end

    % Populate the output struct similar to your original ctt structure
    ctt.xywi = [fitted_x, fitted_y, fitted_Wg, fitted_Ag, R_squared, RMSE];

    % --- Optional: Debugging Plot for Gaussian Fit ---
    if enable_debug_plot
        hFigDebug = figure('Name', 'Gaussian Fit Debug', 'units', 'pixels', 'Position', [100 100 800 400]);
        
        subplot(1, 2, 1);
        imagesc(fit_data);
        colormap(hot);
        axis image;
        title(sprintf('Original ROI (gpxl=%d)', gpxl));
        xlabel('Local X'); ylabel('Local Y');
        colorbar;

        subplot(1, 2, 2);
        % Create the fitted 2D Gaussian surface
        [X_mesh_local, Y_mesh_local] = meshgrid(x_min_fit:x_max_fit, y_min_fit:y_max_fit);
        fitted_surface_data = gaussian2D(p_fit, [X_mesh_local(:), Y_mesh_local(:)]);
        fitted_surface_data = reshape(fitted_surface_data, size(fit_data));
        
        imagesc(fitted_surface_data);
        colormap(hot);
        axis image;
        title(sprintf('Fitted Gaussian (R^2=%.2f, RMSE=%.2f)', R_squared, RMSE));
        xlabel('Local X'); ylabel('Local Y');
        colorbar;
        drawnow;
        % You might want to pause or wait for user input here before closing, or save the figure
        % pause(0.1); % Small pause to see the plot
        % close(hFigDebug); % Close automatically after viewing
    end
end