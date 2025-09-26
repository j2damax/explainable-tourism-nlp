#!/bin/bash

# Set default values
BASE_URL=${BASE_URL:-http://localhost:8000}

# Function to display help
show_help() {
  echo "Usage: ./test_serendip_api.sh [OPTION]..."
  echo "Test the Serendip Experiential Engine API."
  echo ""
  echo "Options:"
  echo "  -u, --url URL           Base URL of the API (default: http://localhost:8000)"
  echo "  -e, --endpoint TYPE     Endpoint to test: health, predict, explain, all (default: all)"
  echo "  -t, --type REVIEW_TYPE  Type of review: eco, wellness, culinary, adventure, mixed (default: all)"
  echo "  -h, --help              Display this help and exit"
  echo ""
  echo "Examples:"
  echo "  ./test_serendip_api.sh                  Test all endpoints with all review types"
  echo "  ./test_serendip_api.sh --endpoint health  Test only the health endpoint"
  echo "  ./test_serendip_api.sh --type eco       Test eco-tourism reviews only"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      show_help
      exit 0
      ;;
    -u|--url)
      BASE_URL="$2"
      shift 2
      ;;
    -e|--endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    -t|--type)
      REVIEW_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Function to make API call and display results
call_api() {
  local endpoint=$1
  local json_data=$2
  local description=$3
  
  echo "===================================================================="
  echo "Testing $description"
  echo "===================================================================="
  
  if [ -z "$json_data" ]; then
    # GET request
    curl -s -X GET "$BASE_URL/$endpoint" | jq .
  else
    # POST request
    echo "Request data: $json_data"
    curl -s -X POST "$BASE_URL/$endpoint" \
      -H "Content-Type: application/json" \
      -d "$json_data" | jq .
  fi
  
  echo ""
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo "Error: jq is not installed. Please install it to format JSON output."
  echo "  macOS: brew install jq"
  echo "  Ubuntu: apt-get install jq"
  exit 1
fi

# Example reviews
ECO_REVIEW='{"review_text": "We loved the sustainable eco-lodge in Ella. The property uses solar power, harvests rainwater, and serves organic food from their garden. The hosts educated us about local conservation efforts and we participated in a tree planting initiative during our stay."}'
WELLNESS_REVIEW='{"review_text": "The Ayurvedic spa retreat near Kandy was transformative. Daily yoga sessions at sunrise, meditation by the lake, and personalized herbal treatments helped me reconnect with myself. I have never felt more balanced and centered after a vacation."}'
CULINARY_REVIEW='{"review_text": "Our cooking class in Galle was incredible! We visited the local market to select spices, learned traditional Sri Lankan curry recipes, and enjoyed our homemade hoppers and sambols. The chef explained the medicinal properties of each spice and herb used."}'
ADVENTURE_REVIEW='{"review_text": "Hiking through Knuckles Mountain Range was challenging but rewarding. We crossed hanging bridges, discovered hidden waterfalls that weren't on any tourist map, and camped under the stars. Our local guide shared stories about the area's unique biodiversity."}'
MIXED_REVIEW='{"review_text": "Our trip to Sri Lanka combined adventure and wellness perfectly. We hiked through tea plantations in the morning, enjoyed farm-to-table Sri Lankan cuisine for lunch, and ended with Ayurvedic treatments. The eco-friendly resort used sustainable practices which we appreciated."}'

# Test health endpoint
if [ -z "$ENDPOINT" ] || [ "$ENDPOINT" = "all" ] || [ "$ENDPOINT" = "health" ]; then
  call_api "health" "" "Health endpoint"
fi

# Test prediction endpoints
if [ -z "$ENDPOINT" ] || [ "$ENDPOINT" = "all" ] || [ "$ENDPOINT" = "predict" ]; then
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "eco" ]; then
    call_api "predict" "$ECO_REVIEW" "Eco-tourism review prediction"
  fi
  
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "wellness" ]; then
    call_api "predict" "$WELLNESS_REVIEW" "Wellness review prediction"
  fi
  
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "culinary" ]; then
    call_api "predict" "$CULINARY_REVIEW" "Culinary review prediction"
  fi
  
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "adventure" ]; then
    call_api "predict" "$ADVENTURE_REVIEW" "Adventure review prediction"
  fi
  
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "mixed" ]; then
    call_api "predict" "$MIXED_REVIEW" "Mixed review prediction"
  fi
fi

# Test explanation endpoints
if [ -z "$ENDPOINT" ] || [ "$ENDPOINT" = "all" ] || [ "$ENDPOINT" = "explain" ]; then
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "eco" ]; then
    call_api "explain" "$ECO_REVIEW" "Eco-tourism review explanation"
  fi
  
  if [ -z "$REVIEW_TYPE" ] || [ "$REVIEW_TYPE" = "all" ] || [ "$REVIEW_TYPE" = "wellness" ]; then
    call_api "explain" "$WELLNESS_REVIEW" "Wellness review explanation"
  fi
fi

echo "All tests completed!"