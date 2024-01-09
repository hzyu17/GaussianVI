# Install headers
file(GLOB ngd "*.h")
install(FILES ${dynamics_headers} DESTINATION include/ngd)