import http from "k6/http";
import { sleep } from 'k6';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

export const options = {
    stages: [
        { duration: '30s', target: 200}, 
        { duration: '10m', target: 200},
        { duration: '5m', target: 0},
    ],
};

export default function() {
    let response = http.get("http://192.168.39.81:30168");

    sleep(randomIntBetween(1, 20)); // sleep between 1 and 5 seconds.
};