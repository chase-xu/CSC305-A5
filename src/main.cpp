// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{
    //TODO: setup uniform contains points?
    float near = -near_plane;
    float far = -far_plane;
    float t = std::abs(near) * std::tan(field_of_view / 2);
    // float t = near * std::tan(field_of_view / 2);
    // float t = camera_top[1];
    float b = -t;
    float r = aspect_ratio * t;
    float l = -r;

    uniform.ortho_proj <<
       2.f / (r - l), 0, 0, -(r+l)/(r-l),
       0, 2.f / (t - b), 0, -(t+b)/(t-b),
       0, 0, 2.f / (near - far), -(near+far)/(near-far),
       0, 0, 0, 1;
    // //TODO: setup camera, compute w, u, v
    Vector3d w = - camera_gaze.normalized();
    Vector3d u = (camera_top.cross(w)).normalized();
    Vector3d v = w.cross(u);

    Matrix4f m_vp;
    int nx = H;
    int ny = H;
    m_vp << nx/2, 0, 0, (nx-1)/2,
        0, ny/2, 0, (ny-1)/2,
        0, 0, 1, 0,
        0, 0, 0, 1; 

    Matrix4f temp;
    temp << u(0), v(0), w(0), camera_position(0), 
    u[1], v[1], w[1], camera_position[1],
    u[2], v[2], w[2], camera_position[2],
    0, 0, 0, 1;
    
    const Matrix4f m_cam =  temp.inverse();
    // //TODO: compute the camera transformation
    uniform.camera =  m_cam;
    Matrix4d P;
    if (is_perspective)
    {
       //TODO setup prespective camera
        uniform.projection = uniform.ortho_proj * m_cam;
    }
    else
    {
        uniform.projection = uniform.ortho_proj;
    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes out;
        out.position = uniform.projection * va.position;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    VertexAttributes v1;
    VertexAttributes v2;
    VertexAttributes v3;
    for (int i =0;i<facets.rows();++i){
        int a = facets(i,0);
        int b = facets(i,1);
        int c = facets(i,2);

        v1 = VertexAttributes(vertices(a,0),vertices(a,1),vertices(a,2));
        v2 = VertexAttributes(vertices(b,0),vertices(b,1),vertices(b,2));
        v3 = VertexAttributes(vertices(c,0),vertices(c,1),vertices(c,2));
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
    }
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

Matrix4f compute_rotation(const double alpha)
{
    //TODO: Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4f res;

    res << cos(alpha), 0, sin(alpha), 0,
        0, 1, 0, 0,
        -sin(alpha), 0, cos(alpha), 0,
        0, 0, 0, 1;

    // rotate around z
    // const double a = alpha;
    // res << cos(a), -sin(a), 0,0,
    //     sin(a), cos(a), 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4f trafo = compute_rotation(alpha);

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes out;
        Vector4f pos =  -uniform.projection * va.trafo * va.position;
        out.position = pos;
        // std::cout<< tmp1<<endl;
        // out.position << tmp1(0,0), tmp1(0, 1), tmp1(0, 2), tmp1(0,3);
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: generate the vertex attributes for the edges and rasterize the lines
    //TODO: use the transformation matrix
    VertexAttributes v1;
    VertexAttributes v2;
    VertexAttributes v3;
    for (int i =0;i<facets.rows();++i){
        int a = facets(i,0);
        int b = facets(i,1);
        int c = facets(i,2);
        //line 1
        v1 = VertexAttributes(vertices(a,0),vertices(a,1),vertices(a,2));
        v2 = VertexAttributes(vertices(b,0),vertices(b,1),vertices(b,2));
        v3 = VertexAttributes(vertices(c,0),vertices(c,1),vertices(c,2));
        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v1);

        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
        //line 3
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v3);
    }


    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}



void get_shading_program(Program &program)
{
    
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: transform the position and the normal
        //TODO: compute the correct lighting
        VertexAttributes out;
        out.position = -uniform.projection * va.trafo * va.position;
        // out.position = -uniform.projection  * va.position;
        Vector3d p, N;
        N = va.norm;
        p << out.position[0], out.position[1], out.position[2];
        Vector3d lights_color(0, 0, 0);
        for (int i = 0; i < light_positions.size(); ++i)
        // for (int i = 0; i < 1; ++i)
        {
            const Vector3d &light_position = light_positions[i]; //light source position of each light
            const Vector3d &light_color = light_colors[i];// light color of one light

            const Vector3d Li = (light_position - p).normalized(); //normal vector pointing to light from intersectio
            const Vector3d shadow_ray =  (light_position - p).normalized();
            Vector3d diff_color = obj_diffuse_color;
            Vector3d specular_color = obj_specular_color;
            // Diffuse contribution
            const Vector3d diffuse = diff_color * std::max(Li.dot(N), 0.0);
            const Vector3d camera_view_position = camera_position - p;
            const Vector3d h = (shadow_ray + (-camera_view_position)).normalized();
            const Vector3d specular = -specular_color * pow(std::max(h.dot(N), 0.0), obj_specular_exponent);
            // std::cout << specular << std::endl;
            // Attenuate lights according to the squared distance to the lights
            const Vector3d D = light_position - p;

            lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
            // lights_color += diffuse + specular;
        }
        Vector3d C = ambient_light + lights_color;

        //Set alpha to 1
        // C(3) = 1;
        out.color << C(0), C(1), C(2), 1;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: create the correct fragment
        Vector4f color = va.color;
        FragmentAttributes out (color[0], color[1], color[2], color[3]);
        out.position = va.position;
        out.depth = va.position[2];
        return out;
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: implement the depth check
        // double alpha = fa.color(3);
        // Vector4d new_color (fa.color(0), fa.color(1), fa.color(2), fa.color(3));
        // Vector4d pre_color (previous.color(0)/255, previous.color(1)/255,previous.color(2)/255,previous.color(3)/255);
        // Vector4d out_color = (1 - alpha)*pre_color + alpha*new_color;
        if (fa.depth < previous.depth)
        {
            // float alpha = fa.color[3];
            // Eigen::Vector4f blend = fa.color.array() * alpha + (previous.color.cast<float>().array() / 255) * (1 - alpha);
            // FrameBufferAttributes out (blend[0] * 255, blend[1] * 255, blend[2] * 255, blend[3] * 255);
            // FrameBufferAttributes out(out_color[0] * 255, out_color[1] * 255, out_color[2] * 255, out_color[3] * 255);
            FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);

            out.depth = fa.depth;
            return out;
        }
        else
            return previous;

    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4f trafo = compute_rotation(alpha);

    std::vector<VertexAttributes> vertex_attributes;
    //TODO: compute the normals
    //TODO: set material colors
    VertexAttributes v1;
    VertexAttributes v2;
    VertexAttributes v3;
    Matrix3d thd_trafo;
    thd_trafo << trafo(0,0), trafo(0,1), trafo(0,2),
                trafo(1,0), trafo(1,1), trafo(1,2),
                trafo(2,0), trafo(2,1), trafo(2,2);

    for (int i =0;i<facets.rows();++i){
        int a = facets(i,0);
        int b = facets(i,1);
        int c = facets(i,2);
        //line 1
        Vector3d k;
        k << vertices(a,0),vertices(a,1),vertices(a,2);
        k = thd_trafo * k;
        Vector3d h;
        h <<vertices(b,0),vertices(b,1),vertices(b,2);
        h = thd_trafo * h;
        Vector3d q;
        q << vertices(c,0),vertices(c,1),vertices(c,2);
        q= thd_trafo * q;
        Vector3d u = k - h;
        Vector3d w = q - h;
        Vector3d norm = u.cross(w).normalized();

        v1 = VertexAttributes(vertices(a,0),vertices(a,1),vertices(a,2));
        v1.norm = norm;
        v2 = VertexAttributes(vertices(b,0),vertices(b,1),vertices(b,2));
        v2.norm = norm;
        v3 = VertexAttributes(vertices(c,0),vertices(c,1),vertices(c,2));
        v3.norm = norm;
        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;

        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);

    }
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4f trafo = compute_rotation(alpha);
    std::vector<VertexAttributes> vertex_attributes;
    //TODO: create vertex attributes
    //TODO: set material colors
    VertexAttributes v1;
    VertexAttributes v2;
    VertexAttributes v3;
    MatrixXd norm_vs (vertices.rows(), 3);
    norm_vs.setZero();
    Matrix3d thd_trafo;
    thd_trafo << trafo(0,0), trafo(0,1), trafo(0,2),
                trafo(1,0), trafo(1,1), trafo(1,2),
                trafo(2,0), trafo(2,1), trafo(2,2);

    for (int i =0;i<facets.rows();++i){
        int a = facets(i,0);
        int b = facets(i,1);
        int c = facets(i,2);
        //line 1
        Vector3d k;
        k << vertices(a,0),vertices(a,1),vertices(a,2);
        k = thd_trafo * k;
        Vector3d h;
        h <<vertices(b,0),vertices(b,1),vertices(b,2);
        h = thd_trafo * h;
        Vector3d q;
        q << vertices(c,0),vertices(c,1),vertices(c,2);
        q = thd_trafo * q;
        Vector3d u = k - h;
        Vector3d w = q - h;
        Vector3d norm = u.cross(w).normalized();
        norm_vs(a,0) += norm(0);
        norm_vs(a, 1) += norm(1);
        norm_vs(a,2)+=norm(2);
        norm_vs(b,0) += norm(0);
        norm_vs(b, 1) += norm(1);
        norm_vs(b,2)+=norm(2);
        norm_vs(c,0) += norm(0);
        norm_vs(c, 1) += norm(1);
        norm_vs(c,2)+=norm(2);

    }

    for(int i =0;i<facets.rows();++i){
        int a = facets(i,0);
        int b = facets(i,1);
        int c = facets(i,2);
        
        v1 = VertexAttributes(vertices(a,0),vertices(a,1),vertices(a,2));
        v1.norm << norm_vs(a, 0), norm_vs(a, 1), norm_vs(a, 2);
        v1.norm = v1.norm.normalized();

        v2 = VertexAttributes(vertices(b,0),vertices(b,1),vertices(b,2));
        v2.norm << norm_vs(b, 0), norm_vs(b, 1), norm_vs(b, 2);
        v2.norm = v2.norm.normalized();

        v3 = VertexAttributes(vertices(c,0),vertices(c,1),vertices(c,2));
        v3.norm << norm_vs(c, 0), norm_vs(c, 1), norm_vs(c, 2);
        v3.norm = v3.norm.normalized();

        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
    }
    // std::cout<< norm_vs<<endl;
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    
    frameBuffer.setConstant(FrameBufferAttributes());
    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());
    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());
    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    //TODO: add the animation
    const char *fileName = "wireframe_rotation.gif";
    int delay = 25;
    GifWriter g;
    GifBegin(&g, fileName, frameBuffer.rows(), frameBuffer.cols(), delay);

    for (float i = 0; i < 2*M_PI; i += 0.3)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        // wireframe_render(i, frameBuffer);
        // flat_shading(i, frameBuffer);
        pv_shading(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    return 0;
}
