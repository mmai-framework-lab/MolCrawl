# NCBI Genome Species Browser - Large Dataset Performance

## Overview

After updating the NCBI genome species list, the dataset grew to 51,273 species. The web interface was redesigned to reliably handle large-scale rendering and browsing.

## Implemented Improvements

### 1. Backend API optimization (`molcrawl-web/api/genome-species.js`)

- In-memory cache with 5-minute retention
- Pagination support (default 1000, max 2000)
- Case-insensitive partial search by species name
- Processing-time and cache-hit tracking
- Efficient batch processing for large files

Core concepts:

- `speciesCache`: `Map`-based cache
- `applyFiltersAndPagination`: filtering and paging logic
- `processingTimeMs`: response latency tracking
- `cacheHits`: cache efficiency tracking

### 2. Frontend virtualized rendering (`molcrawl-web/src/SpeciesBrowser.js`)

- `react-window` (`FixedSizeList`) for virtual scrolling
- 300ms search debounce to reduce API pressure
- User-selectable page size (100 to 2000)
- Responsive layout for desktop and mobile
- Runtime performance indicators (latency/cache status)

Example virtual list settings:

```javascript
<List
  height={400}
  itemCount={species.length}
  itemSize={40}
  overscanCount={5}
>
```

## Performance Characteristics

### Dataset scale

- Total species: **51,273** across 10 categories
- Example categories:
  - Bacteria: 5,184
  - Invertebrates: 8,401
  - Plants: 6,412
  - Vertebrates: 6,477

### Runtime performance

- First load: around 166ms
- Cached requests: around 3-5ms
- Virtual scrolling: smooth 60fps-class interaction

### Memory efficiency

- Renders only visible rows in DOM
- Stable memory behavior for large lists
- Reduced repeated I/O via cache

## API Endpoints

```bash
# Fast stats-only response
GET /api/genome-species?statsOnly=true

# Category with pagination
GET /api/genome-species-category?category=bacteria&limit=100&offset=0

# Category with search
GET /api/genome-species-category?category=plants&search=arabidopsis&limit=50
```

## File Layout

```text
molcrawl-web/
├── api/
│   └── genome-species.js
├── src/
│   ├── SpeciesBrowser.js
│   ├── SpeciesBrowser.css
│   └── App.js
├── species-test.html
└── server.js
```

## Tech Stack

### Backend

- Node.js + Express
- Filesystem-based data loading
- `Map`-based memory cache

### Frontend

- React 19.1.1
- react-window 2.2.0
- CSS Grid/Flexbox

## Before and After

### Previous bottlenecks

- Browser freeze with one-shot rendering of 51k+ rows
- Excessive DOM size and memory pressure
- Slower user interactions under heavy list rendering

### Current behavior

- Virtualized rendering scales with dataset size
- Smooth scrolling and filtering experience
- Cache-backed fast repeated access
- Practical usage on mobile devices

## Operations Notes

### Cache operations

```javascript
speciesCache.clear();
console.log("Cache size:", speciesCache.size);
```

### Runtime headers

```javascript
headers["X-Processing-Time"] = processingTimeMs;
headers["X-Cache-Status"] = cacheUsed ? "HIT" : "MISS";
```

### Error handling

- File read failures
- Memory pressure safeguards
- Network and request-failure handling

## Future Enhancements

1. Export selected species as CSV/JSON
2. Multi-select and batch operations
3. Advanced filtering conditions
4. Real-time sync via WebSocket
5. Distribution analytics and visualization

## Verification

```bash
cd molcrawl-web
node server.js

curl http://localhost:3001/species-test.html
curl "http://localhost:3001/api/genome-species?statsOnly=true"
```

## Summary

The species browser now supports enterprise-scale NCBI species data with a maintainable architecture: virtualized UI, cache-aware APIs, and paginated access.
