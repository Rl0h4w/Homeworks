plugins {
    id("java") // Подключаем плагины для работы с Java-кодом
}

group = "org.example" // Группа проекта. Обычно указывается такой же, как и пакет Main-класса
version = "1.0-SNAPSHOT" // Версия приложения

// Библиотеки хранятся в репозиториях, эта секция подключает репозитории
repositories {
    mavenCentral() // Де-факто стандарт среди репозиториев, самый большой и популярный
}

// Определяем зависимые библиотеки
dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    //Commons Lang
    implementation("org.apache.commons:commons-lang3:3.17.0")
    testImplementation("org.apache.commons:commons-lang3:3.17.0")
}

// Настраиваем задачу по тестированию
tasks.test {
    useJUnitPlatform() // Используем библиотеку JUnit для тестирования
}
