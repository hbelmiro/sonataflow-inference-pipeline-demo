package org.acme.sonataflow.inference.pipeline;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.is;

@QuarkusTest
class InferenceWorkflowIT {

    @Test
    void test() {
        given()
                .contentType(ContentType.JSON)
                .accept(ContentType.JSON)
                .when().post("/inference")
                .then()
                .statusCode(201)
                .body("workflowdata.preProcessingData", is("Data was pre processed"))
                .body("workflowdata.modelServerData", is("modelServer data"))
                .body("workflowdata.postProcessingData", is("Data was post processed"))
                .body("workflowdata.today", is(LocalDate.now().getDayOfMonth()));
    }
}
