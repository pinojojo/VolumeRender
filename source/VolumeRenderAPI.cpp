#include "VolumeRenderAPI.h"

#include "ApplicationVolumeRender.h"

#include <map>

#include <nlohmann/json.hpp>

static int g_instanceCount = 0;
static std::map<int, ApplicationVolumeRender *> g_instanceMap;

void *Init()
{
    auto appDesc = ApplicationDesc{};

    appDesc.Width = 1000;
    appDesc.Height = 800;

    appDesc.Tittle = std::string("Volume Render ") + std::to_string(g_instanceCount++);
    appDesc.IsFullScreen = false;
    appDesc.IsOffscreen = true;

    ApplicationVolumeRender *app = new ApplicationVolumeRender(appDesc);

    g_instanceMap[g_instanceCount] = app;

    return app;
}

int Test()
{
    std::cout << "Test" << std::endl;

    return 3;
}

const char *Status()
{
    nlohmann::json j;

    j["instance_count"] = g_instanceMap.size();

    for (auto const &instance : g_instanceMap)
    {
        j[std::to_string(instance.first)] = instance.second->GetDesc().Tittle;
    }

    return j.dump().c_str();
}

void List()
{
    for (auto const &instance : g_instanceMap)
    {
        std::cout << "Instance " << instance.first << " address is " << instance.second << std::endl;
    }
}

int Render(int w, int h)
{
    if (g_instanceMap.empty())
    {
        return -1;
    }

    g_instanceMap.begin()->second->Run(5);

    return 0;
}

int Cleanup(void *instance)
{
    return 0;
}
