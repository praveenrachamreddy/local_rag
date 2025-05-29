# Use official NGINX image
FROM nginx:latest

# Copy your HTML and config
COPY index.html /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Clean logs and expose port
RUN rm -rf /etc/nginx/conf.d/default.conf && \
    cp /usr/share/nginx/html/index.html /usr/share/nginx/html/

EXPOSE 80

# Run NGINX in foreground
CMD ["nginx", "-g", "daemon off;"]
