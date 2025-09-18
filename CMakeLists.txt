cmake_minimum_required(VERSION 3.20)
project(modular_resources)

find_package(ament_cmake REQUIRED)


install(DIRECTORY json models yaml
        DESTINATION share/${PROJECT_NAME}
)

install(FILES families.yaml
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
