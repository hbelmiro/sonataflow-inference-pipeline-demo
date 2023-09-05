package org.acme.sonataflow.inference.pipeline;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static io.restassured.RestAssured.given;
import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.is;

@QuarkusTest
class InferenceWorkflowTest {

    @Test
    void test() throws IOException {
        given()
                .contentType(ContentType.JSON)
                .accept(ContentType.JSON)
                .when().post("/inference")
                .then()
                .statusCode(201)
                .body("workflowdata.image", is("coco_image.jpg"))
                .body("workflowdata.kserve_response", is("kserve_response.json"))
                .body("workflowdata.model_server_data.segmented_image", is("test_image_house.jpg"))
                .body("workflowdata.postProcessingData", is("Data was post processed"))
                .body("workflowdata.overlaid_image", is("overlaid_image.jpg"));

        Path kserveResponsePayload = Paths.get("kserve_response.json");
        try {
            assertThat(kserveResponsePayload).exists();
        } finally {
            Files.delete(kserveResponsePayload);
        }

        Path kserveRequestPayload = Paths.get("coco_image.json");
        try {
            assertThat(kserveRequestPayload).exists();
        } finally {
            Files.delete(kserveRequestPayload);
        }

        Path overlaidImage = Paths.get("overlaid_image.jpg");
        try {
            assertThat(overlaidImage).exists();
        } finally {
            Files.delete(overlaidImage);
        }
    }
}
