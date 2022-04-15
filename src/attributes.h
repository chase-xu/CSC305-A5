#pragma once

#include <Eigen/Core>

class VertexAttributes
{
public:
    VertexAttributes(float x = 0, float y = 0, float z = 0, float w = 1)
    {
        position << x, y, z, w;
    }

    // Interpolates the vertex attributes
    static VertexAttributes interpolate(
        const VertexAttributes &a,
        const VertexAttributes &b,
        const VertexAttributes &c,
        const float alpha,
        const float beta,
        const float gamma)
    {
        VertexAttributes r;
        r.position = alpha * a.position + beta * b.position + gamma * c.position;
        r.color = alpha * a.color + beta * b.color + gamma * c.color;

        return r;
    }

    Eigen::Vector4f position;
    Eigen::Vector4f color;
    Eigen::Vector3d norm;
    Eigen::Matrix4f trafo;
};

class FragmentAttributes
{
public:
    FragmentAttributes(float r = 0, float g = 0, float b = 0, float a = 1)
    {
        color << r, g, b, a;
    }

    Eigen::Vector4f color;
    Eigen::Vector4f position;
    float depth;
};

class FrameBufferAttributes
{
public:
    FrameBufferAttributes(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
    {
        color << r, g, b, a;
        depth = 2; // this value should be between -1 and 1, 2 is further than  the visible range

    }

    Eigen::Matrix<uint8_t, 4, 1> color;
    float depth;
};

class UniformAttributes
{
public:
    Eigen::Matrix4f projection;
    Eigen::Matrix4f view;
    Eigen::Matrix4f camera;
    Eigen::Matrix4f ortho_proj;
    // Vector4d color;
};