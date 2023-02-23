cmake_minimum_required(VERSION 3.5)
project(modular_resources)

find_package(ament_cmake REQUIRED)


install(DIRECTORY json models yaml
        DESTINATION share/${PROJECT_NAME}
)

ament_package()
