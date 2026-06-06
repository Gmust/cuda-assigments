#include "input_handler.h"
#include <plog/Log.h>
#include <random>

namespace cuda_filter
{

    InputHandler::InputHandler(const FilterOptions &options)
        : m_options(options), m_frameCounter(0), m_isStaticImage(false)
    {
        switch (m_options.inputSource)
        {
        case InputSource::WEBCAM:
            m_cap.open(m_options.deviceId);
            if (!m_cap.isOpened())
            {
                PLOG_ERROR << "Failed to open webcam with device ID: " << m_options.deviceId;
            }
            break;

        case InputSource::IMAGE:
            m_staticFrame = cv::imread(m_options.inputPath);
            if (m_staticFrame.empty())
            {
                PLOG_ERROR << "Failed to load image file: " << m_options.inputPath;
            }
            else
            {
                m_isStaticImage = true;
            }
            break;

        case InputSource::VIDEO:
            m_cap.open(m_options.inputPath);
            if (!m_cap.isOpened())
            {
                PLOG_ERROR << "Failed to open video file: " << m_options.inputPath;
            }
            break;

        case InputSource::SYNTHETIC:
            m_isStaticImage = false;
            break;
        }
    }

    InputHandler::~InputHandler()
    {
        if (m_cap.isOpened())
        {
            m_cap.release();
        }
    }

    bool InputHandler::isOpened() const
    {
        switch (m_options.inputSource)
        {
        case InputSource::WEBCAM:
        case InputSource::VIDEO:
            return m_cap.isOpened();
        case InputSource::IMAGE:
            return !m_staticFrame.empty();
        case InputSource::SYNTHETIC:
            return true;
        default:
            return false;
        }
    }

    bool InputHandler::readFrame(cv::Mat &frame)
    {
        switch (m_options.inputSource)
        {
        case InputSource::WEBCAM:
        case InputSource::VIDEO:
            return m_cap.read(frame);

        case InputSource::IMAGE:
            if (m_isStaticImage)
            {
                frame = m_staticFrame.clone();
                return true;
            }
            return false;

        case InputSource::SYNTHETIC:
            createSyntheticFrame(frame);
            return true;

        default:
            return false;
        }
    }

    void InputHandler::displayFrame(const cv::Mat &frame)
    {
        cv::imshow("Filter Output", frame);
    }

    void InputHandler::displaySideBySide(const cv::Mat &frame1, const cv::Mat &frame2)
    {
        cv::Mat combined;
        cv::hconcat(frame1, frame2, combined);
        cv::imshow("Original vs Filtered", combined);
    }

    void InputHandler::createSyntheticFrame(cv::Mat &frame)
    {
        // Default size for synthetic patterns
        frame.create(480, 640, CV_8UC3);

        switch (m_options.syntheticPattern)
        {
        case SyntheticPattern::CHECKERBOARD:
            createCheckerboard(frame);
            break;
        case SyntheticPattern::GRADIENT:
            createGradient(frame);
            break;
        case SyntheticPattern::NOISE:
            createNoise(frame);
            break;
        }

        m_frameCounter++;
    }

    void InputHandler::createCheckerboard(cv::Mat &frame)
    {
        const int squareSize = 40;
        const int numSquaresX = frame.cols / squareSize;
        const int numSquaresY = frame.rows / squareSize;

        for (int y = 0; y < frame.rows; y++)
        {
            for (int x = 0; x < frame.cols; x++)
            {
                int squareX = x / squareSize;
                int squareY = y / squareSize;
                bool isBlack = (squareX + squareY + m_frameCounter) % 2 == 0;

                uchar value = isBlack ? 0 : 255;
                frame.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
            }
        }
    }

    void InputHandler::createGradient(cv::Mat &frame)
    {
        for (int y = 0; y < frame.rows; y++)
        {
            for (int x = 0; x < frame.cols; x++)
            {
                float phase = (x + y + m_frameCounter) * 0.1f;
                uchar value = static_cast<uchar>(128 + 127 * sin(phase));

                frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    value,              // Blue
                    (value + 85) % 256, // Green (shifted)
                    (value + 170) % 256 // Red (shifted)
                );
            }
        }
    }

    void InputHandler::createNoise(cv::Mat &frame)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 255);

        for (int y = 0; y < frame.rows; y++)
        {
            for (int x = 0; x < frame.cols; x++)
            {
                frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    dis(gen), // Blue
                    dis(gen), // Green
                    dis(gen)  // Red
                );
            }
        }
    }

} // namespace cuda_filter