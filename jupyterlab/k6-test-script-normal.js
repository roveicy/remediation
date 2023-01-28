import http from "k6/http";
import { sleep } from 'k6';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';
import { normalDistributionStages } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

export const options = {
    // Alters the number of VUs from 1 to 10 over a period
    // of 20 seconds comprised of 5 stages.
    stages: normalDistributionStages(200, 900, 5),
  };
  

export default function() {
    let response = http.get("http://192.168.39.81:30168");

    sleep(randomIntBetween(1, 5)); // sleep between 1 and 5 seconds.
};