#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// === Helper Functions === //

string clean_path(string path) {
    
    // Removes Extra Whitespace if detected.

    path.erase(0, path.find_first_not_of(" \t\r\n\""));
    path.erase(path.find_last_not_of(" \t\r\n\"") + 1);

    return path;
}

bool is_number(const string& str) {
    if(str.empty()) return false;
    string::const_iterator iterator = str.begin();
    while(iterator !=str.end()){
        if(isdigit(*iterator))
            iterator++;
        else return false;
    }
    return true;

}

void get_dataset_names(vector<pair<string, string>> & dataset_name_path) {
    string dataset_path = "Dataset";
    for(const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            for(const auto& inner_entry: fs::directory_iterator(entry.path())) {
                if(inner_entry.is_regular_file()){
                    string name, path;
                    name = path = entry.path().string();
                    name.erase(0, 8); // erase Dataset 
                    replace(name.begin(), name.end(), '\\', ' '); // replace backslash
                    dataset_name_path.push_back({name, path});
                    break;
                }
            }            
        }
    }
}

// === Algorithm Steps === //

bool set_directories(string& input_directory, string& output_directory) {
    
        string input;
        // == Dataset Names == //
        vector<pair<string,string>> dataset_name_path;
        get_dataset_names(dataset_name_path);

        cout << "Please Choose A Dataset For Your Inputes Frames Directory: \n";
        for(int i = 0; i < dataset_name_path.size(); i++){
            cout << i + 1 << ") "<< dataset_name_path[i].first << "\n";
        }
        
        // == Input Directory == //
          cout << "Or Enter The Path Of Your Inputes Frames Directory: \n";

          getline(cin, input);
          input = clean_path(input);

          if(is_number(input)) {
              // Validating dataset number.
              int number = stoi(input) - 1;
              if (number < 0 || number >= dataset_name_path.size()) {
                  cerr << "Error: Input number does not exist or is not a valid number.\n";
                  return 0;
              }
              input_directory  = dataset_name_path[stoi(input) - 1].second;
              output_directory = "Output\\" + dataset_name_path[stoi(input) - 1].first;
          } else
              input_directory = input;
      

  // Validating selected input directory.
  if (!fs::exists(input_directory) || !fs::is_directory(input_directory)) {
      cerr << "Error: Input directory does not exist or is not a valid folder.\n";
      return 0;
  }


        // == Output Directory == //
  if(output_directory.empty()){
      cout << "Please Enter The Desired output Path.: \n";
      getline(cin, output_directory);
      output_directory = clean_path(output_directory);
  }
 




  // Validating selected output directory (+ Creating it if it doesn't exist.) 
  if (!fs::exists(output_directory)) {
      if (!fs::create_directory(output_directory)) {
          cerr << "Error: Could not create output directory!" << endl;
          return 0;
      }
  }
  return 1; // no error
}

bool load_images(vector<pair<string, Mat>>& input_images, const string& input_directory) {
    try {
        for (const auto& entry : fs::directory_iterator(input_directory)) {
            if (entry.is_regular_file()) {
                Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
                if (img.empty()) {
                    cerr << "Warning: Could not read image: " << entry.path() << endl;
                    continue;
                }
                input_images.push_back({ entry.path().filename().string(), img });
                cout << "Loaded: " << entry.path() << endl;
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        return 0;
    }

    if (input_images.empty()) {
        cerr << "Error: No images found in directory." << endl;
        return 0;
    }
    return 1; // no error
}

bool set_frames(int& nFrames_selected, const int& input_images_size){
    cout << "How Many Frames would you like to process? (Min: 1, Max: " << input_images_size << "): ";
	cin >> nFrames_selected;

    if (nFrames_selected <= 0 || nFrames_selected > input_images_size)
    {
		cerr << "Error: Invalid number of frames selected. Exiting." << endl;
        return 0;
    }
    return 1;
}

bool validate_images_same_size(const vector<Mat>& images) {
    if (images.empty()) {
        cerr << "Error: No images to validate!" << endl;
        return false;
    }

    int standard_rows = images[0].rows;
    int standard_cols = images[0].cols;

    for (size_t i = 1; i < images.size(); ++i) {
        if (images[i].rows != standard_rows || images[i].cols != standard_cols) {
            cerr << "Error: Image at index " << i << " does not match size ("
                << standard_rows << "x" << standard_cols << "). "
                << "Found (" << images[i].rows << "x" << images[i].cols << ")." << endl;
            return false;
        }
    }

    cout << "All images have consistent size: "
        << standard_rows << "x" << standard_cols << "." << endl;
    return true;
}

Mat compute_estimated_background(const vector<Mat>& images) {
    
    if (images.empty()) {
        cerr << "Error: No images found!" << endl;
        return Mat(); // Return empty matrix
    }

    int num_images = images.size(); // Total number of images (M)
    // Get the size of images
    int rows = images[0].rows; // Number of rows (height)
    int cols = images[0].cols; // Number of columns (width)
    
    // Create a matrix to accumulate sum of pixel values, We use floating point to avoid integer overflow
    Mat background_sum = Mat::zeros(rows, cols, CV_32FC1);

    // Loop over all images one by one
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {

        // Get current image
        Mat current_image = images[img_idx];

        // Loop over all pixels of the current image
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {

                // Get pixel value at (row, col) from current image
                uchar pixel_value = current_image.at<uchar>(row, col);

                // Convert to float for safe addition
                float pixel_value_float = static_cast<float>(pixel_value);

                // Add pixel value to the accumulator
                background_sum.at<float>(row, col) += pixel_value_float;
            }
        }
    }

    //Create a matrix to store the final background (float type first)
    Mat background_mean = Mat::zeros(rows, cols, CV_32FC1);
    float total_sum = 0;
    float mean_value = 0;
    // Loop over each pixel again to divide sum by number of images (M)
    // B(row,col) = (pixel1(row,col) + pixel2(row,col) + ... + pixelM(row,col)) / M
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {

            // Get the accumulated sum at (row, col)
            total_sum = background_sum.at<float>(row, col);

            // Calculate mean (average) value
             mean_value = total_sum / static_cast<float>(num_images);

            // Store the mean value in the background_mean matrix
            background_mean.at<float>(row, col) = mean_value;
        }
    }

    // Convert the background from float to 8-bit for display/saving
    Mat background_final;
    background_mean.convertTo(background_final, CV_8UC1);

    //Return the computed background image
    return background_final;
}

bool extract_images_for_processing(vector<Mat>& images_only, const vector<pair<string, Mat>>& input_images, const int& nFrames_selected){
    try {
        for (int i = 0; i < nFrames_selected; i++) {
            images_only.push_back(input_images[i].second);
        }
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 0;
    }
    return 1;

}

void save_display_background(const string& output_directory, const Mat& background){
    imwrite(output_directory + "\\estimated_background.jpg", background);
    imshow("Estimated Background", background);
    waitKey(500); // Show for a short time
}

Mat compute_foreground_mask(const Mat& background, const Mat& frame, int threshold_value) {
    if (background.empty()) {
        cerr << "Error: Background image is empty!" << endl;
        return Mat();
    }
    if (frame.empty()) {
        cerr << "Error: Input frame image is empty!" << endl;
        return Mat();
    }

    //Get the size of the images
    int rows = background.rows;
    int cols = background.cols;

    // Create a black image for the foreground mask (initially all pixels = 0)
    Mat foreground_mask = Mat::zeros(rows, cols, CV_8UC1);

    // Loop over every pixel (row by row, column by column)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {

            // Read pixel values at (row, col)
            int background_pixel_value = static_cast<int>(background.at<uchar>(row, col));
            int frame_pixel_value = static_cast<int>(frame.at<uchar>(row, col));

            // Compute absolute difference between background and current frame
            int pixel_difference = abs(background_pixel_value - frame_pixel_value);

            // Compare the difference with threshold
            if (pixel_difference > threshold_value) {
                // If the difference is greater than threshold, mark it as foreground (white pixel)
                foreground_mask.at<uchar>(row, col) = 255;
            }
            else {
                // Else, mark it as background (black pixel)
                foreground_mask.at<uchar>(row, col) = 0;
            }
        }
    }

    // Return the computed foreground mask
    return foreground_mask;
}

void save_foreground(const vector<Mat>& images_only, const string& output_directory, const Mat& background){
    int threshold_value = 59;
    int counter = 0;

    for (size_t i = 0; i < images_only.size(); ++i) {
        const Mat& frame = images_only[i];
        Mat foreground_mask = compute_foreground_mask(background, frame, threshold_value);
        if (!foreground_mask.empty()) {
            string output_filename = output_directory + "\\foreground_mask_" + to_string(counter) + ".jpg";
            imwrite(output_filename, foreground_mask);
            cout << "Saved: " << output_filename << endl;
            counter++;

            // Optional: Display each foreground mask
            imshow("Foreground Mask", foreground_mask);
            waitKey(300); // Short delay
        }
    }

    cout << "All foreground masks saved into " << output_directory << "!" << endl;

    system(("start \"\" \"" + output_directory + "\"").c_str());
}

int main() {

    // === Receiving Directories === //
    string input_directory, output_directory;
    if(set_directories(input_directory, output_directory) == 0)
        return -1;


    vector<pair<string, Mat>> input_images;
    // === Loading Images ===
    if(load_images(input_images, input_directory) == 0)
        return -1;
    else
        cout << "Successfully loaded " << input_images.size() << " images." << endl;


    // === Decide Number Of Frames To Be Processed === //
    int nFrames_selected = 0;
    if(set_frames(nFrames_selected, input_images.size()) == 0)
        return -1;
  

    // === Extract Images for Processing ===
    vector<Mat> images_only;
    if(extract_images_for_processing(images_only, input_images, nFrames_selected) == 0)
        return -1;


    // === Validate All Images Have the Same Size ===
    if (!validate_images_same_size(images_only)) {
        cerr << "Error: Images are not the same size. Cannot continue." << endl;
        return -1;
    }


    // === Compute Background Image Manually ===
    Mat background = compute_estimated_background(images_only);
    if (background.empty()) {
        cerr << "Background computation failed!" << endl;
        return -1;
    }


    // === Save and Display the Estimated Background ===
    save_display_background(output_directory, background);


    // === Compute and Save Foreground Masks Manually ===
    save_foreground(images_only, output_directory, background);

    return 0;
}
