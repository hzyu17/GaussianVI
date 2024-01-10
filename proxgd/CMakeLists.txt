# Install headers
file(GLOB proxgd "*.h")
install(FILES ${dynamics_headers} DESTINATION include/proxgd)