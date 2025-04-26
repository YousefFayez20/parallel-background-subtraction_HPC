#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

bool validate_image_sizes(const vector<Mat>& images) {
    if (images.empty()) return false;
    int width = images[0].cols;
    int height = images[0].rows;
    for (size_t i = 1; i < images.size(); ++i) {
        if (images[i].cols != width || images[i].rows != height) {
            cerr << "Error: Image " << i << " has a different size!" << endl;
            return false;
        }
    }
    return true;
}

Mat compute_background(const vector<Mat>& images) {
    if (images.empty()) {
        cerr << "No images to process!" << endl;
        return Mat();
    }

    Mat accumulator = Mat::zeros(images[0].size(), CV_32FC1);

    for (const auto& img : images) {
        Mat temp;
        img.convertTo(temp, CV_32FC1);
        accumulator += temp;
    }

    accumulator /= static_cast<float>(images.size());

    Mat background;
    accumulator.convertTo(background, CV_8UC1);

    return background;
}

Mat compute_foreground_mask(const Mat& background, const Mat& frame, int threshold_value) {
    Mat diff, mask;
    absdiff(background, frame, diff);
    threshold(diff, mask, threshold_value, 255, THRESH_BINARY);
    return mask;
}

int main() {
    string input_directory = "D:\\HPC\\Images";
    string output_directory = "D:\\HPC\\ForegroundResults"; // NEW output folder

    vector<pair<string, Mat>> input_images;

    try {
        for (const auto& entry : fs::directory_iterator(input_directory)) {
            if (entry.is_regular_file()) {
                string file_path = entry.path().string();
                Mat img = imread(file_path, IMREAD_GRAYSCALE);
                if (img.empty()) {
                    cerr << "Warning: Could not read image: " << file_path << endl;
                    continue;
                }
                input_images.push_back({ entry.path().filename().string(), img });
                cout << "Loaded: " << file_path << endl;
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        return -1;
    }

    if (input_images.empty()) {
        cerr << "Error: No images found in directory." << endl;
        return -1;
    }

    cout << "Successfully loaded " << input_images.size() << " images." << endl;

    // Extract just Mat images for background calculation
    vector<Mat> images_only;
    for (const auto& pair : input_images) {
        images_only.push_back(pair.second);
    }

    if (!validate_image_sizes(images_only)) {
        cerr << "Error: Not all images have the same size." << endl;
        return -1;
    }

    // === Compute Background ===
    Mat background = compute_background(images_only);
    if (background.empty()) {
        cerr << "Background computation failed!" << endl;
        return -1;
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(output_directory)) {
        if (!fs::create_directory(output_directory)) {
            cerr << "Error: Could not create output directory!" << endl;
            return -1;
        }
    }

    // Save background into output folder
    imwrite(output_directory + "\\estimated_background.jpg", background);
    imshow("Estimated Background", background);
    waitKey(500); // Show for a short time

    // === Compute Foreground for every frame  ===
    int threshold_value = 58;
    int counter = 0;

    for (const auto& [filename, frame] : input_images) {
        Mat foreground_mask = compute_foreground_mask(background, frame, threshold_value);
        if (!foreground_mask.empty()) {
            string output_filename = output_directory + "\\foreground_mask_" + to_string(counter) + ".jpg";
            imwrite(output_filename, foreground_mask);
            cout << "Saved: " << output_filename << endl;
            counter++;

            // Optional: Display each mask
            imshow("Foreground Mask", foreground_mask);
            waitKey(300);
        }
    }

    cout << "All foreground masks saved into " << output_directory << "!" << endl;

    return 0;
}