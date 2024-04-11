# Stage 0, "build-stage", based on Node.js, to build and compile the frontend
FROM node:lts-gallium as build-stage

WORKDIR /app

COPY package*.json /app/

RUN npm install

COPY ./ /app/

RUN npm run build

# Stage 1, based on Nginx, to have only the compiled app, ready for production with Nginx
FROM nginx:1.23-alpine

COPY --from=build-stage /app/dist/qafront/ /usr/share/nginx/html

# Copy nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

#trigger