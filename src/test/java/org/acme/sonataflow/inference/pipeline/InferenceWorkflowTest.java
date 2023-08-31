package org.acme.sonataflow.inference.pipeline;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;

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
                .body("workflowdata.preProcessingData", is("Data was pre processed"))
                .body("workflowdata.modelServerData", is("modelServer data"))
                .body("workflowdata.postProcessingData", is("Data was post processed"))
                .body("workflowdata.today", is(LocalDate.now().getDayOfMonth()))
                .body("workflowdata.overlaid_image", is("overlaid_image.jpg"));

        Path overlaidImage = Paths.get("overlaid_image.jpg");
        try {
            assertThat(overlaidImage).exists();
        } finally {
            Files.delete(overlaidImage);
        }
    }
}
