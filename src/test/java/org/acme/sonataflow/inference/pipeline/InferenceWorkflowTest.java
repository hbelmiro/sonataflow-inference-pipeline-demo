package org.acme.sonataflow.inference.pipeline;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.config.HttpClientConfig;
import io.restassured.config.RestAssuredConfig;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import javax.xml.bind.DatatypeConverter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.is;

@QuarkusTest
class InferenceWorkflowTest {

    private static final int REQUEST_TIMEOUT = 90_000;

    @Test
    void test() throws IOException {
        String imageBase64;

        try (InputStream imageBase64Stream = getClass().getResourceAsStream("/coco_image.jpg")) {
            imageBase64 = DatatypeConverter.printBase64Binary(Objects.requireNonNull(imageBase64Stream).readAllBytes());
        }

        String outputImageBase64;
        try (InputStream outputImageBase64Stream = getClass().getResourceAsStream("/output.png")) {
            outputImageBase64 = DatatypeConverter.printBase64Binary(Objects.requireNonNull(outputImageBase64Stream).readAllBytes());
        }

        given().config(configRequest())
                .contentType(ContentType.JSON)
                .accept(ContentType.JSON)
                .body("{\"base64Image\" : \"" + imageBase64 + "\"}").when()
                .when().post("/pipeline")
                .then()
                .statusCode(201)
                .body("workflowdata.output_image", is(outputImageBase64));
    }

    private static RestAssuredConfig configRequest() {
        return RestAssuredConfig.config()
                .httpClient(HttpClientConfig.httpClientConfig()
                        .setParam("http.socket.timeout", REQUEST_TIMEOUT)
                        .setParam("http.connection.timeout", REQUEST_TIMEOUT));
    }
}
