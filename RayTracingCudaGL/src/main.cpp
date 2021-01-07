#include "main.hpp"

using namespace std;

//Sphere** list;
//Sphere** world;
//Camera** camera;

// GLFW
GLFWwindow* window;
int WIDTH  = 600;
int HEIGHT = 300;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLProgram* shdrawtex = new GLSLProgram();

// Cuda <-> OpenGl interop resources
void* cuda_dev_render_buffer; // Cuda buffer for initial render
struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result

// CUDA
size_t size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions          // Colors           // Texture Coords
	1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
	1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
	-1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
	-1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
	0, 1, 3,  // First Triangle
	1, 2, 3   // Second Triangle
};

// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y)
{
	// create an OpenGL texture
	glGenTextures(1, gl_tex); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target

	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	SDK_CHECK_ERROR_GL();
}

void initGLBuffers()
{
	// create texture that will receive the result of cuda kernel
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, WIDTH, HEIGHT);

	// create shader program
	try
	{
		shdrawtex->compileShader("./Assets/Shaders/OpaqueShader.vert");
		shdrawtex->compileShader("./Assets/Shaders/OpaqueShader.frag");

		shdrawtex->link();
		shdrawtex->use();
	}
	catch (GLSLProgramException& e)
	{
		cerr << e.what() << endl;
		exit(EXIT_FAILURE);
	}

	SDK_CHECK_ERROR_GL();
}

bool initGL(){
	glewExperimental = GL_TRUE; // need this to enforce core profile
	GLenum err = glewInit();
	glGetError(); // parse first error
	if (err != GLEW_OK) {// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s /n", glewGetErrorString(err));
		exit(1);
	}
	glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
	SDK_CHECK_ERROR_GL();
	return true;
}

void initCUDABuffers()
{
	// set up vertex data parameters
	num_texels = WIDTH * HEIGHT;
	num_values = num_texels * 4; // 4 because of RGBA
	size_tex_data = sizeof(GLubyte) * num_values; // GLubyte -> 8 unsigned integer

	// We don't want to use cudaMallocManaged here - since we definitely want
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
}

bool initGLFW(){
	if (!glfwInit()) 
		exit(EXIT_FAILURE);
	
	// These hints switch the OpenGL profile to core
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "RayTracing Cuda/GL", NULL, NULL);

	if(!window)
	{ 
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, Input::KeyCallback);

	return true;
}

void generateCUDAImage()
{
	// calculate grid size
	int tx = 8;
	int ty = 8;

	dim3 blocks(WIDTH/tx+1, HEIGHT/ty+1); // 2D grid, every thread will compute a pixel
	dim3 threads(tx, ty);

	launch_cudaRender(blocks, threads, (unsigned int *) cuda_dev_render_buffer, WIDTH, HEIGHT); // launch with 0 additional shared memory allocated

	// We want to copy cuda_dev_render_buffer data to the texture
	// Map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	int num_texels = WIDTH * HEIGHT;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
}

void display(void) 
{
	generateCUDAImage();

	glfwPollEvents();
	// Clear the color buffer
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

  // we gonna use this compiled GLSL program
	shdrawtex->use();

	glUniform1i(glGetUniformLocation(shdrawtex->getHandle(), "tex"), 0);

	glBindVertexArray(VAO); // binding VAO automatically binds EBO
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0); // unbind VAO

	SDK_CHECK_ERROR_GL();
	
	// Swap the screen buffers
	glfwSwapBuffers(window);
}

int main(int argc, char *argv[]) 
{
	initGLFW();
	initGL();

	printGLFWInfo(window);
	printGlewInfo();
	printGLInfo();

	findCudaDevice(argc, (const char **)argv);
	initGLBuffers();
	initCUDABuffers();
	
	// Generate world
	create_world(WIDTH, HEIGHT, WIDTH * HEIGHT);

	// Generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Buffer setup
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute (3 floats)
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	// Texture attribute (2 floats)
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.
	while (!glfwWindowShouldClose(window))
	{
		display();
		glfwWaitEvents();
	}

	free_world();

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
