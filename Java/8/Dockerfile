FROM gradle:8.5-jdk17 AS build

WORKDIR /home/gradle/project

COPY --chown=gradle:gradle . /home/gradle/project

RUN gradle build --no-daemon

FROM openjdk:17-jdk-slim

WORKDIR /app

COPY --from=build /home/gradle/project/app/build/libs/*.jar /app/app.jar

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
