#ifndef PC_APPLICATION_H
#define PC_APPLICATION_H

#include "Application.h"


class PCApplication : public Application
{
public:
	PCApplication(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras);

	bool BInitGL(uint8_t* colorData, uint8_t* depthData);
	bool HandleUserInput();
	bool BInitGLTotal_cyf(uint8_t* colorData, uint8_t* depthData);
	bool BInit_cyf();
	bool BInitGLPreCyf();
	bool BInitGL_framebuffers_cyf();

	// keep track of user input
	glm::vec3 accumMovement = glm::vec3();
	glm::vec3 accumRotation = glm::vec3();
};

PCApplication::PCApplication(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras)
	: Application(options, fpsMonitor, inputCameras, outputCameras){};

// cyf
bool PCApplication::BInit_cyf()
{
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	int nWindowPosX = 20;
	int nWindowPosY = 20;
	// Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
	Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN;

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	//SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	m_pCompanionWindow = SDL_CreateWindow("OpenDIBR", nWindowPosX, nWindowPosY, options.SCR_WIDTH, options.SCR_HEIGHT, unWindowFlags);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		printf("%s - Error initializing GLEW! %s\n", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	if (SDL_GL_SetSwapInterval(0) < 0) // 0 means vsync off, 1 means vsync on
	{
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	std::string strWindowTitle = "OpenDIBR";
	SDL_SetWindowTitle(m_pCompanionWindow, strWindowTitle.c_str());

	if (!BInitGL_framebuffers_cyf())
	{
		printf("%s - Unable to initialize framebuffers!\n", __FUNCTION__);
		return false;
	}

	return true;
}

bool PCApplication::BInitGL(uint8_t* colorData, uint8_t* depthData)
{
	if (!Application::BInitGL(colorData, depthData)) {
		return false;
	}
	auto timeA = std::chrono::high_resolution_clock::now();

	framebuffers.init(inputCameras, options.SCR_WIDTH, options.SCR_HEIGHT, options);
    auto timeB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = timeB - timeA;
	std::cout << "PCApplication::BInitGL time: " << elapsed1.count() << " ms\n";

	return true;
}

// cyf
bool PCApplication::BInitGLPreCyf()
{
	printf("222222\n");

	if (!Application::BInitGLPreCyf()) {
		return false;
	}
	printf("PCApplication::BInitGLPreCyf()\n");
	return true;
}

bool PCApplication::BInitGLTotal_cyf(uint8_t* colorData, uint8_t* depthData)
{
	if (!Application::BInitGL(colorData, depthData)) {
		return false;
	}
	printf("PCApplication::BInitGLTotal_cyf()\n");
	return true;
}

bool PCApplication::BInitGL_framebuffers_cyf()
{
	framebuffers.init(inputCameras, options.SCR_WIDTH, options.SCR_HEIGHT, options);
	return true;
}

bool PCApplication::HandleUserInput() 
{
	SDL_Event sdlEvent;
	bool bRet = false;

	glm::vec3 movement(0);
	glm::vec3 rotation(0);

	while (SDL_PollEvent(&sdlEvent) != 0)
	{
		if (sdlEvent.type == SDL_QUIT)
		{
			bRet = true;
		}
		else if (sdlEvent.type == SDL_KEYDOWN)
		{
			if (sdlEvent.key.keysym.sym == SDLK_ESCAPE)
			{
				bRet = true;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_v)
			{
				cameraSpeed += cameraSpeed * 0.1f;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_c)
			{
				cameraSpeed = std::max(cameraSpeed * 0.9f, 0.001f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_n)
			{
				options.blendingFactor = std::min(options.blendingFactor + 1, 10);
				std::cout << "changed blending_factor to " << options.blendingFactor << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("blendingThreshold", 0.001f + options.blendingFactor * 0.004f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_b)
			{
				options.blendingFactor = std::max(options.blendingFactor -1, 0);
				std::cout << "changed blending_factor to " << options.blendingFactor << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("blendingThreshold", 0.001f + options.blendingFactor * 0.004f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_h)
			{
				options.triangle_deletion_margin += 2;
				std::cout << "changed triangle_deletion_margin to " << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_g)
			{
				options.triangle_deletion_margin = std::max(options.triangle_deletion_margin - 2, 1.0f);
				std::cout << "changed triangle_deletion_margin to " << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (options.showCameraVisibilityWindow && sdlEvent.key.keysym.sym == SDLK_r) {
				controlCameraVisibilityWindow = !controlCameraVisibilityWindow;
				if (controlCameraVisibilityWindow) std::cout << "now controlling the small window in the bottom right corner" << std::endl;
				else std::cout << "now controlling the main window" << std::endl;
			}
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONDOWN && sdlEvent.button.button == SDL_BUTTON_LEFT) {
			leftMouseDown = true;
			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONUP && sdlEvent.button.button == SDL_BUTTON_LEFT)
		{
			leftMouseDown = false;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONDOWN && sdlEvent.button.button == SDL_BUTTON_MIDDLE) {
			middleMouseDown = true;
			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONUP && sdlEvent.button.button == SDL_BUTTON_MIDDLE)
		{
			middleMouseDown = false;
		}
		else if (leftMouseDown && sdlEvent.type == SDL_MOUSEMOTION)
		{
			float deltaX = (float)sdlEvent.motion.x - prev_mouse_pos_x;
			float deltaY = (float)sdlEvent.motion.y - prev_mouse_pos_y;

			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;

			if (controlCameraVisibilityWindow) {
				cameraVisibilityWindow.angle -= deltaX * 0.005f;
			}
			else {
				// rotation
				rotation.y = -deltaX * 0.002f;
				rotation.x = -deltaY * 0.002f;
			}
		}
		else if (middleMouseDown && sdlEvent.type == SDL_MOUSEMOTION)
		{
			float deltaX = (float)sdlEvent.motion.x - prev_mouse_pos_x;
			float deltaY = (float)sdlEvent.motion.y - prev_mouse_pos_y;

			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;

			if (controlCameraVisibilityWindow) {
				cameraVisibilityWindow.angle -= deltaX * 0.005f;
			}
			else {
				// translation
				movement.x = -deltaX * 0.05f * cameraSpeed;
				movement.y = deltaY * 0.05f * cameraSpeed;
			}
		}
		else if (sdlEvent.type == SDL_MOUSEWHEEL)
		{
			if (sdlEvent.wheel.y > 0) // scroll up
			{
				if (controlCameraVisibilityWindow) {
					cameraVisibilityWindow.radius += 0.05f * abs(cameraVisibilityWindow.radius);
				}
				else {
					// translation
					movement.z -= cameraSpeed * 6;
				}
			}
			else if (sdlEvent.wheel.y < 0) // scroll down
			{
				if (controlCameraVisibilityWindow) {
					cameraVisibilityWindow.radius -= 0.05f * abs(cameraVisibilityWindow.radius);
				}
				else {
					// translation
					movement.z += cameraSpeed * 6;
				}
			}
		}
	}

	const Uint8* state = SDL_GetKeyboardState(NULL);
	if (state[SDL_SCANCODE_W]) { // UP
		movement.y += cameraSpeed;
	}
	if (state[SDL_SCANCODE_S]) { // DOWN
		movement.y -= cameraSpeed;
	}
	if (state[SDL_SCANCODE_D]) { // RIGHT
		movement.x += cameraSpeed;
	}
	if (state[SDL_SCANCODE_A]) { // LEFT
		movement.x -= cameraSpeed;
	}
	if (state[SDL_SCANCODE_Z]) { // BACKWARD
		movement.z += cameraSpeed;
	}
	if (state[SDL_SCANCODE_Q]) { // FORWARD
		movement.z -= cameraSpeed;
	}

	if (movement != glm::vec3(0) || rotation != glm::vec3(0)) {

		accumRotation += rotation;
		glm::mat4 inputRx = glm::rotate(glm::mat4(1.0f), accumRotation[0], glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 inputRy = glm::rotate(glm::mat4(1.0f), accumRotation[1], glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 inputRz = glm::rotate(glm::mat4(1.0f), accumRotation[2], glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 rotMat = inputRz * inputRy * inputRx;
		accumMovement += glm::mat3(pcOutputCamera.startRotMat * rotMat) * movement;
		glm::mat4 posMat = glm::translate(glm::mat4(1.0f), pcOutputCamera.pos + accumMovement);
		pcOutputCamera.model = posMat * pcOutputCamera.startRotMat * rotMat;
		pcOutputCamera.view = glm::inverse(pcOutputCamera.model);

	}

	return bRet;
}


#endif PC_APPLICATION_H