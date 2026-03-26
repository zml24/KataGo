#include "../dataio/numpywrite.h"

#include <cstring>

#ifndef NO_LIBZIP
#include <zip.h>
#endif

using namespace std;

#if !defined(BYTE_ORDER) || (BYTE_ORDER != LITTLE_ENDIAN && BYTE_ORDER != BIG_ENDIAN)
#error Define BYTE_ORDER to be equal to either LITTLE_ENDIAN or BIG_ENDIAN
#endif

template <>
NumpyBuffer<uint8_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|u1")
{}
template <>
NumpyBuffer<int8_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|i1")
{}
template <>
NumpyBuffer<bool>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|b1")
{}

#if BYTE_ORDER == LITTLE_ENDIAN
template <>
NumpyBuffer<float>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<f4")
{}
template <>
NumpyBuffer<double>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<f8")
{}
template <>
NumpyBuffer<uint16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u2")
{}
template <>
NumpyBuffer<int16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i2")
{}
template <>
NumpyBuffer<uint32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u4")
{}
template <>
NumpyBuffer<int32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i4")
{}
template <>
NumpyBuffer<uint64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u8")
{}
template <>
NumpyBuffer<int64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i8")
{}
#else
template <>
NumpyBuffer<float>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">f4")
{}
template <>
NumpyBuffer<double>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">f8")
{}
template <>
NumpyBuffer<uint16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u2")
{}
template <>
NumpyBuffer<int16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i2")
{}
template <>
NumpyBuffer<uint32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u4")
{}
template <>
NumpyBuffer<int32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i4")
{}
template <>
NumpyBuffer<uint64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u8")
{}
template <>
NumpyBuffer<int64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i8")
{}
#endif

template <typename T>
NumpyBuffer<T>::NumpyBuffer(const vector<int64_t>& shp, const char* dt)
  : shape(shp),dtype(dt)
{
  dataLen = 1;
  assert(shape.size() > 0);
  for(size_t i = 0; i<shape.size(); i++) {
    assert(shape[i] >= 0);
    if((uint64_t)dataLen * (uint64_t)shape[i] < (uint64_t)dataLen)
      throw StringError("NumpyBuffer shape overflows");
    dataLen *= shape[i];
  }

  //Leave 256 bytes at the start for the header
  int sizeOfT = sizeof(T);
  assert(sizeOfT > 0 && sizeOfT <= TOTAL_HEADER_BYTES);

  headerLen = TOTAL_HEADER_BYTES / sizeOfT;
  assert(headerLen * sizeOfT == TOTAL_HEADER_BYTES);

  dataIncludingHeader = new T[headerLen+dataLen];
  data = dataIncludingHeader + headerLen;

  //Go ahead and write all the magic bytes and such
  char* s = (char*)dataIncludingHeader;
  s[0] = (char)0x93;
  s[1] = 'N';
  s[2] = 'U';
  s[3] = 'M';
  s[4] = 'P';
  s[5] = 'Y';
  s[6] = 0x1;
  s[7] = 0x0;
  //Remaining bytes in header = 246 past these two bytes.
  s[8] = (char)((TOTAL_HEADER_BYTES - 10) & 0xFF);
  s[9] = (char)((TOTAL_HEADER_BYTES - 10) >> 8);

  string dictStrFirstHalf = Global::strprintf(
    "{'descr':'%s','fortran_order':False,'shape':(",
    dt
  );

  if(dictStrFirstHalf.size() > TOTAL_HEADER_BYTES - 40)
    throw StringError("Numpy header dict is too long, datatype string is too long: " + string(dt));
  strcpy(s+10, dictStrFirstHalf.c_str());

  //Record where we should start writing the shape and finish off the dict
  shapeStartByte = dictStrFirstHalf.size() + 10;
}

template <typename T>
NumpyBuffer<T>::~NumpyBuffer() {
  delete[] dataIncludingHeader;
}

template <typename T>
int64_t NumpyBuffer<T>::getActualDataLen(int64_t numWriteableRows) {
  int64_t actualDataLen = 1;
  for(size_t i = 0; i<shape.size(); i++) {
    int64_t x = (i == 0) ? numWriteableRows : shape[i];
    actualDataLen *= x;
  }
  return actualDataLen;
}

//Writes the header of the buffer and returns the total size of the writeable portion of
//the buffer, in bytes.
//Writes the header and computes the size treating the writeable length of the leading dimension
//of the shape to be just numRows rather than the specified size at creation time.
//This is so that users can preallocate one buffer at the start and still write it
//if there were not as many rows as expected ("partial batch").
template <typename T>
uint64_t NumpyBuffer<T>::prepareHeaderWithNumRows(int64_t numWriteableRows) {
  //Continue writing the shape
  size_t idx = shapeStartByte;
  char* s = (char*)dataIncludingHeader;

  //Write each number
  int64_t actualDataLen = 1;
  for(size_t i = 0; i<shape.size(); i++) {
    if(i > 0) {
      s[idx] = ',';
      idx += 1;
      if(idx >= TOTAL_HEADER_BYTES)
        throw StringError("Numpy header is too long, datatype and shape are too long");
    }

    int64_t x = (i == 0) ? numWriteableRows : shape[i];
    actualDataLen *= x;

    int numDigits = 0;
    char digitsRev[32];
    if(x == 0) {
      digitsRev[0] = '0';
      numDigits = 1;
    }
    else {
      while(x > 0) {
        digitsRev[numDigits++] = '0' + (x % 10);
        x /= 10;
      }
    }

    for(int j = numDigits-1; j >= 0; j--) {
      s[idx] = digitsRev[j];
      idx += 1;
      if(idx >= TOTAL_HEADER_BYTES)
        throw StringError("Numpy header is too long, datatype and shape are too long");
    }
  }
  //Finish
  s[idx] = ')'; //close tuple for shape
  idx += 1;
  if(idx >= TOTAL_HEADER_BYTES)
    throw StringError("Numpy header is too long, datatype and shape are too long");
  s[idx] = '}'; //close dict literal
  idx += 1;
  if(idx >= TOTAL_HEADER_BYTES)
    throw StringError("Numpy header is too long, datatype and shape are too long");

  //Pad with spaces
  while(idx < TOTAL_HEADER_BYTES-1) {
    s[idx] = ' ';
    idx += 1;
  }
  s[idx] = '\n'; //newline, as specified by numpy
  idx += 1;

  return (uint64_t)(TOTAL_HEADER_BYTES + actualDataLen * sizeof(T));
}

template struct NumpyBuffer<float>;
template struct NumpyBuffer<double>;
template struct NumpyBuffer<bool>;
template struct NumpyBuffer<uint8_t>;
template struct NumpyBuffer<uint16_t>;
template struct NumpyBuffer<uint32_t>;
template struct NumpyBuffer<uint64_t>;
template struct NumpyBuffer<int8_t>;
template struct NumpyBuffer<int16_t>;
template struct NumpyBuffer<int32_t>;
template struct NumpyBuffer<int64_t>;

#ifdef NO_LIBZIP

// Built-in zip writer using store mode (no compression), no external dependencies.
// Produces valid .npz files readable by numpy.
// Limitations: ZIP32 only, maximum total file size < 4 GiB.

#include <fstream>
#include <mutex>
#include <vector>

static const uint64_t ZIP32_MAX = 0xFFFFFFFFULL;

// CRC32 slice-by-8 lookup tables (standard polynomial 0xEDB88320).
// Thread-safe: initialized exactly once via std::call_once.
static uint32_t crc32Tables[8][256];
static std::once_flag crc32TablesOnce;

static void initCrc32Tables() {
  for(uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for(int j = 0; j < 8; j++) {
      if(c & 1)
        c = 0xEDB88320u ^ (c >> 1);
      else
        c >>= 1;
    }
    crc32Tables[0][i] = c;
  }
  for(uint32_t i = 0; i < 256; i++) {
    uint32_t c = crc32Tables[0][i];
    for(int t = 1; t < 8; t++) {
      c = crc32Tables[0][c & 0xFF] ^ (c >> 8);
      crc32Tables[t][i] = c;
    }
  }
}

static uint32_t computeCrc32(const void* data, uint64_t len) {
  std::call_once(crc32TablesOnce, initCrc32Tables);
  const uint8_t* buf = (const uint8_t*)data;
  uint32_t crc = 0xFFFFFFFFu;
  while(len >= 8) {
    uint32_t a = crc ^ ((uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                         ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24));
    uint32_t b = (uint32_t)buf[4] | ((uint32_t)buf[5] << 8) |
                 ((uint32_t)buf[6] << 16) | ((uint32_t)buf[7] << 24);
    crc = crc32Tables[7][a & 0xFF] ^
          crc32Tables[6][(a >> 8) & 0xFF] ^
          crc32Tables[5][(a >> 16) & 0xFF] ^
          crc32Tables[4][(a >> 24) & 0xFF] ^
          crc32Tables[3][b & 0xFF] ^
          crc32Tables[2][(b >> 8) & 0xFF] ^
          crc32Tables[1][(b >> 16) & 0xFF] ^
          crc32Tables[0][(b >> 24) & 0xFF];
    buf += 8;
    len -= 8;
  }
  while(len--) {
    crc = crc32Tables[0][(crc ^ *buf++) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

struct BuiltinZipEntry {
  string name;
  uint32_t crc;
  uint64_t size;
  uint64_t localHeaderOffset;
};

struct BuiltinZipState {
  ofstream out;
  vector<BuiltinZipEntry> entries;
  bool closed;
};

static void zipWriteU16(ofstream& out, uint16_t v) {
  char buf[2];
  buf[0] = (char)(v & 0xFF);
  buf[1] = (char)(v >> 8);
  out.write(buf, 2);
}

static void zipWriteU32(ofstream& out, uint32_t v) {
  char buf[4];
  buf[0] = (char)(v & 0xFF);
  buf[1] = (char)((v >> 8) & 0xFF);
  buf[2] = (char)((v >> 16) & 0xFF);
  buf[3] = (char)(v >> 24);
  out.write(buf, 4);
}

static void checkStream(ofstream& out, const string& fileName, const char* context) {
  if(!out.good())
    throw StringError(string("I/O error ") + context + " zip file " + fileName);
}

ZipFile::ZipFile(const string& fName)
  :fileName(fName),file(NULL)
{
  BuiltinZipState* state = new BuiltinZipState();
  state->closed = false;
  state->out.open(fileName, std::ios::binary | std::ios::trunc);
  if(!state->out.is_open()) {
    delete state;
    throw StringError("Could not open zip file for writing: " + fileName);
  }
  file = state;
}

ZipFile::~ZipFile() {
  BuiltinZipState* state = (BuiltinZipState*)file;
  if(state != NULL) {
    if(state->out.is_open())
      state->out.close();
    delete state;
  }
}

void ZipFile::writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes) {
  BuiltinZipState* state = (BuiltinZipState*)file;

  if(numBytes > ZIP32_MAX)
    throw StringError(
      "Built-in zip writer only supports ZIP32 (< 4 GiB per entry). Entry " +
      string(nameWithinZip) + " is " + to_string(numBytes) + " bytes in " + fileName
    );

  BuiltinZipEntry entry;
  entry.name = nameWithinZip;
  entry.crc = computeCrc32(data, numBytes);
  entry.size = numBytes;
  entry.localHeaderOffset = (uint64_t)state->out.tellp();

  if(entry.localHeaderOffset > ZIP32_MAX)
    throw StringError(
      "Built-in zip writer only supports ZIP32 (< 4 GiB total). File offset exceeded limit writing " +
      string(nameWithinZip) + " in " + fileName
    );

  uint16_t nameLen = (uint16_t)entry.name.size();

  // Local file header (30 fixed bytes + name)
  ofstream& out = state->out;
  zipWriteU32(out, 0x04034b50);  // local file header signature
  zipWriteU16(out, 20);          // version needed to extract (2.0)
  zipWriteU16(out, 0);           // general purpose bit flag
  zipWriteU16(out, 0);           // compression method: store
  zipWriteU16(out, 0);           // last mod file time
  zipWriteU16(out, 0);           // last mod file date
  zipWriteU32(out, entry.crc);
  zipWriteU32(out, (uint32_t)entry.size);   // compressed size
  zipWriteU32(out, (uint32_t)entry.size);   // uncompressed size
  zipWriteU16(out, nameLen);
  zipWriteU16(out, 0);           // extra field length
  out.write(entry.name.c_str(), nameLen);
  out.write((const char*)data, numBytes);

  checkStream(out, fileName, "writing entry to");

  state->entries.push_back(entry);
}

void ZipFile::close() {
  BuiltinZipState* state = (BuiltinZipState*)file;
  if(state->closed) return;

  ofstream& out = state->out;
  uint64_t centralDirOffset = (uint64_t)out.tellp();

  if(centralDirOffset > ZIP32_MAX)
    throw StringError(
      "Built-in zip writer only supports ZIP32 (< 4 GiB total). Central directory offset exceeded limit in " + fileName
    );

  // Central directory
  for(const auto& entry : state->entries) {
    uint16_t nameLen = (uint16_t)entry.name.size();
    zipWriteU32(out, 0x02014b50);  // central directory file header signature
    zipWriteU16(out, 20);          // version made by
    zipWriteU16(out, 20);          // version needed to extract
    zipWriteU16(out, 0);           // general purpose bit flag
    zipWriteU16(out, 0);           // compression method: store
    zipWriteU16(out, 0);           // last mod file time
    zipWriteU16(out, 0);           // last mod file date
    zipWriteU32(out, entry.crc);
    zipWriteU32(out, (uint32_t)entry.size);   // compressed size
    zipWriteU32(out, (uint32_t)entry.size);   // uncompressed size
    zipWriteU16(out, nameLen);
    zipWriteU16(out, 0);           // extra field length
    zipWriteU16(out, 0);           // file comment length
    zipWriteU16(out, 0);           // disk number start
    zipWriteU16(out, 0);           // internal file attributes
    zipWriteU32(out, 0);           // external file attributes
    zipWriteU32(out, (uint32_t)entry.localHeaderOffset);
    out.write(entry.name.c_str(), nameLen);
  }

  checkStream(out, fileName, "writing central directory to");

  uint64_t centralDirEnd = (uint64_t)out.tellp();
  uint64_t centralDirSize = centralDirEnd - centralDirOffset;

  // End of central directory record
  zipWriteU32(out, 0x06054b50);
  zipWriteU16(out, 0);           // number of this disk
  zipWriteU16(out, 0);           // disk where central directory starts
  zipWriteU16(out, (uint16_t)state->entries.size());
  zipWriteU16(out, (uint16_t)state->entries.size());
  zipWriteU32(out, (uint32_t)centralDirSize);
  zipWriteU32(out, (uint32_t)centralDirOffset);
  zipWriteU16(out, 0);           // comment length

  out.flush();
  checkStream(out, fileName, "flushing");

  out.close();
  if(out.fail())
    throw StringError("I/O error closing zip file " + fileName);

  state->closed = true;
}

#else

struct ZipError {
  zip_error_t value;
  ZipError() { zip_error_init(&value); }
  ~ZipError() { zip_error_fini(&value); }
  ZipError(const ZipError&) = delete;
  ZipError& operator=(const ZipError&) = delete;
};

ZipFile::ZipFile(const string& fName)
  :fileName(fName),file(NULL)
{
  ZipError zipError;
  zip_source_t* zipFileSource = zip_source_file_create(fileName.c_str(),0,-1,&(zipError.value));
  if(zipFileSource == NULL)
    throw StringError("Could not open zip file " + fileName + " due to error " + zip_error_strerror(&(zipError.value)));
  zip_t* fileHandle = zip_open_from_source(zipFileSource, ZIP_CREATE | ZIP_TRUNCATE, &(zipError.value));
  file = fileHandle;
  if(file == NULL) {
    zip_source_free(zipFileSource);
    throw StringError("Could not open zip file " + fileName + " due to error " + zip_error_strerror(&(zipError.value)));
  }
}

ZipFile::~ZipFile() {
  if(file != NULL)
    zip_discard((zip_t*)file);
}

void ZipFile::writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes) {
  ZipError zipError;
  zip_source_t* dataSource = zip_source_buffer((zip_t*)file,data,numBytes,0);
  if(dataSource == NULL)
    throw StringError(
      "Could not initialize zip write data buffer for " + string(nameWithinZip) +
      " within " + fileName + " due to error " + zip_error_strerror(&(zipError.value))
    );

  zip_int64_t idx = zip_file_add((zip_t*)file, nameWithinZip, dataSource, ZIP_FL_OVERWRITE);
  if(idx < 0) {
    zip_source_free(dataSource);
    throw StringError(
      "Could not write to " + string(nameWithinZip) +
      " within zip file " + fileName + " due to error " + zip_strerror((zip_t*)file)
    );
  }
}

void ZipFile::close() {
  int result = zip_close((zip_t*)file);
  if(result < 0)
    throw StringError("Could not close zip file " + fileName + " due to error " + zip_strerror((zip_t*)file));
  else
    file = NULL;
}

#endif

// void test() {
//   string fileName = "abc.npz";

//   NumpyBuffer<float> np({4,3,4});
//   for(int i = 0; i<2*3*4; i++)
//     np.data[i] = 0.1*i;

//   uint64_t npBytes = np.prepareHeaderWithNumRows(2);

//   ZipFile zipFile(fileName);
//   zipFile.writeBuffer("nptest",np.dataIncludingHeader,npBytes);
//   zipFile.close();
// }
