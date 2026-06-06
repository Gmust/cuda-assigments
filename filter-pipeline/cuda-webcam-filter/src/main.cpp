#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/filter_utils.h"
#include "kernels/kernels.h"
#include "pipeline/filter_pipeline.h"
#include "pipeline/pipeline_stage.h"

namespace
{
    // Real-time visualization of the pipeline stage timings, drawn as a
    // semi-transparent panel with one horizontal bar per stage (length ∝ GPU
    // ms). Also shows FPS, upload/compute/download split, stream mode, the
    // chain description, and the wipe-transition progress.
    void drawTimingOverlay(cv::Mat &canvas, const cuda_filter::FilterPipeline &pipe,
                           double fps, bool multiStream, int numStreams)
    {
        const auto &timings = pipe.lastStageTimings();
        int panelW = 360;
        int rowH   = 22;
        int panelH = 96 + static_cast<int>(timings.size()) * rowH + 28;

        cv::Mat roi = canvas(cv::Rect(8, 8, std::min(panelW, canvas.cols - 16),
                                      std::min(panelH, canvas.rows - 16)));
        cv::Mat shade(roi.size(), roi.type(), cv::Scalar(20, 20, 20));
        cv::addWeighted(shade, 0.55, roi, 0.45, 0.0, roi);

        auto put = [&](const std::string &s, int x, int y, cv::Scalar col,
                       double scale = 0.45) {
            cv::putText(canvas, s, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                        scale, col, 1, cv::LINE_AA);
        };

        int x = 20, y = 30;
        char buf[160];
        snprintf(buf, sizeof(buf), "FPS %.0f   total %.2f ms   %s",
                 fps, pipe.lastTotalMs(),
                 multiStream ? ("multi x" + std::to_string(numStreams)).c_str()
                             : "single-stream");
        put(buf, x, y, cv::Scalar(0, 255, 255), 0.5);
        y += 22;
        snprintf(buf, sizeof(buf), "H2D %.2f  compute %.2f  D2H %.2f ms",
                 pipe.lastUploadMs(), pipe.lastComputeMs(), pipe.lastDownloadMs());
        put(buf, x, y, cv::Scalar(200, 200, 200));
        y += 20;
        put(cuda_filter::describePipeline(pipe.stages()), x, y,
            cv::Scalar(180, 255, 180));
        y += 24;

        // per-stage bars
        float maxMs = 0.01f;
        for (auto &t : timings) maxMs = std::max(maxMs, t.ms);
        int barX = x + 110, barMax = panelW - 130;
        for (auto &t : timings) {
            put(t.name, x, y + 12, cv::Scalar(255, 255, 255));
            int w = static_cast<int>(barMax * (t.ms / maxMs));
            cv::rectangle(canvas, cv::Rect(barX, y, std::max(2, w), 14),
                          cv::Scalar(60, 200, 90), cv::FILLED);
            snprintf(buf, sizeof(buf), "%.2f", t.ms);
            put(buf, barX + barMax + 6, y + 12, cv::Scalar(220, 220, 220), 0.4);
            y += rowH;
        }

        // transition progress bar
        if (pipe.transitioning()) {
            y += 6;
            put("wipe", x, y + 12, cv::Scalar(255, 180, 80));
            cv::rectangle(canvas, cv::Rect(barX, y, barMax, 12),
                          cv::Scalar(90, 90, 90), 1);
            cv::rectangle(canvas, cv::Rect(barX, y,
                          static_cast<int>(barMax * pipe.transitionProgress()), 12),
                          cv::Scalar(80, 180, 255), cv::FILLED);
        }
    }

    int runPipelineMode(cuda_filter::InputHandler &inputHandler,
                        const cuda_filter::FilterOptions &options)
    {
        using namespace cuda_filter;

        std::vector<Stage> specA = parsePipelineSpec(options.pipelineSpec);
        std::vector<Stage> specB = parsePipelineSpec(options.transitionSpec);

        FilterPipeline pipe;
        pipe.setStages(specA);
        pipe.setMultiStream(options.multiStream);
        pipe.setNumStreams(options.numStreams);

        bool showingA = true;     // tracks which preset we last transitioned toward

        PLOG_INFO << "=== Pipeline mode ===";
        PLOG_INFO << "Controls: ESC quit | m multi/single | [ ] streams-/+ | "
                     "t wipe transition | b/s/e/o/g/i/p add blur/sharpen/edge/emboss/"
                     "gray/invert/sepia | x remove last | c clear";

        cv::Mat frame, output;
        double fps = 0.0;
        int frameCount = 0;
        double fpsClock = static_cast<double>(cv::getTickCount());
        double prevTick = fpsClock;

        while (true)
        {
            if (!inputHandler.readFrame(frame)) {
                PLOG_ERROR << "Failed to read frame";
                break;
            }
            if (frame.channels() == 1)
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

            double now = static_cast<double>(cv::getTickCount());
            float dt = static_cast<float>((now - prevTick) / cv::getTickFrequency());
            prevTick = now;
            pipe.update(dt);

            pipe.process(frame, output);

            frameCount++;
            if ((now - fpsClock) / cv::getTickFrequency() >= 0.5) {
                fps = frameCount / ((now - fpsClock) / cv::getTickFrequency());
                frameCount = 0;
                fpsClock = now;
            }

            drawTimingOverlay(output, pipe, fps, pipe.multiStream(), pipe.numStreams());

            if (options.preview)
                inputHandler.displaySideBySide(frame, output);
            else
                inputHandler.displayFrame(output);

            int key = cv::waitKey(1);
            if (key == 27) break;                          // ESC
            switch (key) {
                case 'm': pipe.setMultiStream(!pipe.multiStream()); break;
                case '[': pipe.setNumStreams(pipe.numStreams() - 1); break;
                case ']': pipe.setNumStreams(pipe.numStreams() + 1); break;
                case 'b': pipe.addStage(makeStage("blur"));    break;
                case 's': pipe.addStage(makeStage("sharpen")); break;
                case 'e': pipe.addStage(makeStage("edge"));    break;
                case 'o': pipe.addStage(makeStage("emboss"));  break;
                case 'g': pipe.addStage(makeStage("grayscale")); break;
                case 'i': pipe.addStage(makeStage("invert"));  break;
                case 'p': pipe.addStage(makeStage("sepia"));   break;
                case 'x': pipe.removeLast(); break;
                case 'c': pipe.clear();      break;
                case 't':
                    if (!pipe.transitioning()) {
                        pipe.beginTransition(showingA ? specB : specA,
                                             options.transitionDuration,
                                             options.wipeSoftness);
                        showingA = !showingA;
                    }
                    break;
                default: break;
            }
        }
        PLOG_INFO << "Pipeline mode terminated";
        return 0;
    }
} // namespace

int main(int argc, char **argv)
{
    // Initialize logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Parse command line arguments
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    // Initialize input handler
    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    // Pipeline mode: a chain of filters with CUDA streams, runtime add/remove,
    // wipe transitions, and real-time timing visualization.
    if (!options.pipelineSpec.empty())
        return runPipelineMode(inputHandler, options);

    // Create filter kernel / HDR options
    cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(options.filterType);
    const bool isHDR = (filterType == cuda_filter::FilterType::HDR_TONEMAPPING);

    cv::Mat kernel;
    cuda_filter::HdrOptions hdrOpts{};
    if (isHDR)
    {
        hdrOpts = {options.exposure, options.gamma, options.saturation,
                   options.whitePoint, options.hdrAlgorithm};
        PLOG_INFO << "Filter: hdr  algorithm=" << options.hdrAlgorithm
                  << "  exposure=" << options.exposure
                  << "  gamma=" << options.gamma
                  << "  saturation=" << options.saturation
                  << "  white-point=" << options.whitePoint;
    }
    else
    {
        kernel = cuda_filter::FilterUtils::createFilterKernel(
            filterType, options.kernelSize, options.intensity);
        PLOG_INFO << "Filter: " << options.filterType
                  << ", Kernel size: " << options.kernelSize
                  << ", Intensity: " << options.intensity;
    }

    cv::Mat frame, filteredCPU, filteredGPU;
    double fpsCPU = 0.0, fpsGPU = 0.0;
    int frameCountCPU = 0, frameCountGPU = 0;
    double startTimeCPU = static_cast<double>(cv::getTickCount());
    double startTimeGPU = static_cast<double>(cv::getTickCount());

    PLOG_INFO << "Press 'ESC' to exit";

    while (true)
    {
        // Capture frame
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        // Apply filter using CPU
        const double cpuStart = static_cast<double>(cv::getTickCount());
        if (isHDR)
            cuda_filter::applyHDRFilterCPU(frame, filteredCPU, hdrOpts);
        else
            cuda_filter::applyFilterCPU(frame, filteredCPU, kernel);
        const double cpuEnd = static_cast<double>(cv::getTickCount());
        const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
        frameCountCPU++;
        if ((cpuEnd - startTimeCPU) / cv::getTickFrequency() >= 1.0)
        {
            fpsCPU = frameCountCPU;
            frameCountCPU = 0;
            startTimeCPU = cpuEnd;
        }

        // Apply filter using GPU
        const double gpuStart = static_cast<double>(cv::getTickCount());
        if (isHDR)
            cuda_filter::applyHDRFilterGPU(frame, filteredGPU, hdrOpts);
        else
            cuda_filter::applyFilterGPU(frame, filteredGPU, kernel);
        const double gpuEnd = static_cast<double>(cv::getTickCount());
        const double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
        frameCountGPU++;
        if ((gpuEnd - startTimeGPU) / cv::getTickFrequency() >= 1.0)
        {
            fpsGPU = frameCountGPU;
            frameCountGPU = 0;
            startTimeGPU = gpuEnd;
        }

        // Add FPS and processing time text to the frames
        std::string cpuText = "CPU FPS: " + std::to_string(static_cast<int>(fpsCPU)) +
                              " Time: " + std::to_string(cpuTime * 1000).substr(0, 4) + "ms";
        std::string gpuText = "GPU FPS: " + std::to_string(static_cast<int>(fpsGPU)) +
                              " Time: " + std::to_string(gpuTime * 1000).substr(0, 4) + "ms";

        cv::putText(filteredCPU, cpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        cv::putText(filteredGPU, gpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        // Create a combined image showing both results
        cv::Mat combined;
        cv::hconcat(filteredCPU, filteredGPU, combined);
        cv::putText(combined, "CPU Version", cv::Point(10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        cv::putText(combined, "GPU Version", cv::Point(combined.cols / 2 + 10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        // Display the combined result
        if (options.preview)
        {
            inputHandler.displaySideBySide(frame, combined);
        }
        else
        {
            inputHandler.displayFrame(combined);
        }

        // Exit on ESC key
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
